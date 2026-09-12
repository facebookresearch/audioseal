# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import urllib

import pytest
import torch
import torchaudio

from audioseal import AudioSeal
from audioseal.models import AudioSealDetector, AudioSealWM, MsgProcessor

if sys.version_info >= (3, 10):
    from audioseal.libs.moshi.modules.seanet import SEANetDecoder, SEANetEncoder
    from audioseal.libs.moshi.modules.streaming import StreamingModule
    from audioseal.libs.moshi.utils.compile import no_compile


@pytest.fixture
def example_audio(tmp_path):
    url = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
    with open(tmp_path / "test.wav", "wb") as f:
        resp = urllib.request.urlopen(url)
        f.write(resp.read())

    wav, sr = torchaudio.load(tmp_path / "test.wav")

    # Add batch dimension
    yield wav.expand(2, 1, -1), sr


def test_detector(example_audio):
    audio, sr = example_audio
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    model.eval()

    secret_message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
    watermark = model.get_watermark(audio, sample_rate=sr, message=secret_message)

    watermarked_audio = audio + watermark

    detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
    detector.eval()
    results, message = detector.detect_watermark(
        watermarked_audio, sample_rate=sr
    )  # noqa

    # Due to non-deterministic decoding, messages are not always the same as message
    print(f"\nOriginal message: {secret_message}")
    print(f"Decoded message: {message}")
    print(
        "Matching bits in decoded and original messages: "
        f"{torch.count_nonzero(torch.eq(message, secret_message)).item()}\n"
    )
    assert torch.count_nonzero(torch.eq(message, secret_message)).item() > 20
    assert torch.all(results > 0.5).item()

    # Try to detect the unwatermarked audio
    results, _ = detector.detect_watermark(audio, sample_rate=sr)  # noqa
    assert torch.all(results < 0.5).item()


def test_loading_from_hf():
    generator = AudioSeal.load_generator(
        "facebook/audioseal/generator_base.pth", nbits=16
    )
    detector = AudioSeal.load_detector("facebook/audioseal/detector_base.pth", nbits=16)

    assert isinstance(generator, AudioSealWM) and isinstance(
        detector, AudioSealDetector
    )


def test_jit(example_audio, tmp_path):
    # Test that the audioseal model can be torchscripted
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    model.eval()
    scripted_model = torch.jit.script(model)
    assert isinstance(scripted_model, torch.jit.ScriptModule)
    scripted_model.save(tmp_path / "audioseal_wm_16bits.jit")
    del scripted_model

    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    scripted_detector = torch.jit.script(detector)
    scripted_detector.save(tmp_path / "audioseal_detector_16bits.jit")
    del scripted_detector

    jit_generator = torch.jit.load(tmp_path / "audioseal_wm_16bits.jit")
    jit_detector = torch.jit.load(tmp_path / "audioseal_detector_16bits.jit")
    audio, _ = example_audio

    wm_audio = jit_generator(audio, alpha=0.8)
    result, _ = jit_detector.detect_watermark(wm_audio)
    assert torch.all(result > 0.5).item(), "JIT model failed to detect watermark"


@pytest.fixture
def streaming_model():
    if sys.version_info < (3, 10):
        pytest.skip("Streaming requires Python 3.10 or later")
    kwargs = dict(
        dimension=8,
        n_filters=2,
        ratios=[8, 5, 4, 2],
        lstm=2,
        causal=True,
        norm="none",
        pad_mode="constant",
    )
    # Keep both model initialization and test inputs reproducible without
    # changing the random state of other tests.
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = AudioSealWM(
            SEANetEncoder(**kwargs),
            SEANetDecoder(**kwargs),
            MsgProcessor(nbits=16, hidden_size=8),
        ).eval()
        yield model


@pytest.fixture
def streaming_generator(streaming_model):
    # Compare state and numerical behavior independently of compilation.
    with no_compile():
        yield streaming_model


@pytest.fixture
def compiled_streaming_generator(streaming_model):
    # Earlier model tests compile the same forward methods with other shapes.
    torch._dynamo.reset()
    try:
        yield streaming_model
    finally:
        torch._dynamo.reset()


@pytest.mark.skipif(
    bool(os.environ.get("NO_TORCH_COMPILE")),
    reason="Compilation disabled by NO_TORCH_COMPILE",
)
@torch.no_grad()
def test_streaming_compiled(compiled_streaming_generator):
    model = compiled_streaming_generator
    audio = torch.randn(2, 1, 960)
    message = torch.zeros(2, 16, dtype=torch.int32)
    with no_compile():
        expected = model(audio, message=message)
    chunks = audio.split(320, dim=-1)
    with model.streaming(2):
        outputs = [model(chunks[0], message=message)]
        saved = model.get_streaming_state()
    with model.streaming(2):
        model.set_streaming_state(saved)
        outputs.extend(model(chunk, message=message) for chunk in chunks[1:])
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1), expected, atol=1e-5, rtol=1e-5
    )
    with model.streaming(2):
        with pytest.raises(ValueError, match="positive multiple"):
            model(audio[..., :321], message=message)
        actual = model(chunks[0], message=message)
    torch.testing.assert_close(actual, outputs[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("chunk_size", [320, 640, 1600])
@torch.no_grad()
def test_streaming_matches_whole_audio(streaming_generator, chunk_size):
    model = streaming_generator
    audio = torch.randn(2, 1, 3200)
    message = torch.zeros(2, 16, dtype=torch.int32)
    expected = model(audio, message=message)
    with model.streaming(2):
        actual = torch.cat(
            [
                model(chunk, message=message)
                for chunk in audio.split(chunk_size, dim=-1)
            ],
            dim=-1,
        )
        for network in (model.encoder, model.decoder):
            modules = [
                module
                for module in network.modules()
                if isinstance(module, StreamingModule)
            ]
            assert all(module.is_streaming() for module in modules)
            states = network.get_streaming_state()
            assert len(states) == len(modules)
            assert any(
                getattr(state, "previous_h", None) is not None
                for state in states.values()
            )
            assert any(
                getattr(state, "previous", None) is not None
                for state in states.values()
            )
        assert any(
            getattr(state, "partial", None) is not None
            for state in model.decoder.get_streaming_state().values()
        )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert all(
        not module.is_streaming()
        for module in model.modules()
        if isinstance(module, StreamingModule)
    )


@torch.no_grad()
def test_streaming_state_restoration(streaming_generator):
    model = streaming_generator
    audio = torch.randn(2, 1, 1280)
    message = torch.ones(2, 16, dtype=torch.int32)
    with model.streaming(2):
        expected = torch.cat(
            [model(chunk, message=message) for chunk in audio.split(320, dim=-1)],
            dim=-1,
        )
    saved = None
    outputs = []
    for chunk in audio.split(320, dim=-1):
        # Exercise an unrelated stream between chunks of the saved request.
        with model.streaming(2):
            model(-audio, message=message)
        with model.streaming(2):
            if saved is not None:
                model.set_streaming_state(saved)
            outputs.append(model(chunk, message=message))
            saved = model.get_streaming_state()
    torch.testing.assert_close(torch.cat(outputs, dim=-1), expected, atol=0, rtol=0)


@pytest.mark.parametrize("length", [0, 321, 1000])
@torch.no_grad()
def test_streaming_rejects_incomplete_frames(streaming_generator, length):
    model = streaming_generator
    valid = torch.randn(1, 1, model.frame_size)
    message = torch.zeros(1, 16, dtype=torch.int32)
    expected = model(valid, message=message)
    with model.streaming(1):
        with pytest.raises(ValueError, match="positive multiple"):
            model(torch.zeros(1, 1, length), message=message)
        actual = model(valid, message=message)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("remainder", [1, 40, 319])
@torch.no_grad()
def test_streaming_final_frame(streaming_generator, remainder):
    model = streaming_generator
    audio = torch.randn(1, 1, 640 + remainder)
    message = torch.zeros(1, 16, dtype=torch.int32)
    # Match the caller's explicit waveform padding, rather than the separate
    # per-layer padding used by non-streaming inference on a partial frame.
    padded = torch.nn.functional.pad(audio, (0, model.frame_size - remainder))
    expected = model(padded, message=message)[..., : audio.shape[-1]]
    with model.streaming(1):
        first = model(audio[..., :640], message=message)
        tail = torch.nn.functional.pad(
            audio[..., 640:], (0, model.frame_size - remainder)
        )
        last = model(tail, message=message)[..., :remainder]
    torch.testing.assert_close(
        torch.cat([first, last], dim=-1), expected, atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Streaming requires Python 3.10 or later"
)
@torch.no_grad()
def test_streaming_pretrained(example_audio):
    audio, sr = example_audio
    audio = torchaudio.functional.resample(audio, sr, 16000)[..., :6400]
    model = AudioSeal.load_generator("audioseal_wm_streaming").eval()
    message = torch.zeros(audio.shape[0], 16, dtype=torch.int32)
    with no_compile():
        expected = model(audio, message=message)
        with model.streaming(audio.shape[0]):
            actual = torch.cat(
                [
                    model(chunk, message=message)
                    for chunk in audio.split(model.frame_size, dim=-1)
                ],
                dim=-1,
            )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    detector = AudioSeal.load_detector("audioseal_detector_streaming").eval()
    with no_compile():
        scores, decoded = detector.detect_watermark(actual)
    assert torch.all(scores > 0.5)
    # Match the existing detector test's tolerance for non-deterministic decoding.
    assert torch.count_nonzero(torch.eq(decoded, message)).item() > 20


def test_streaming_state_requires_both_networks(streaming_generator):
    model = streaming_generator
    with pytest.raises(RuntimeError, match="not streaming"):
        model.get_streaming_state()
    with pytest.raises(RuntimeError, match="not streaming"):
        model.set_streaming_state({})
    with model.streaming(1):
        with pytest.raises(ValueError, match="both encoder and decoder"):
            model.set_streaming_state({"encoder": model.encoder.get_streaming_state()})
        with pytest.raises(RuntimeError, match="already streaming"):
            with model.streaming(1):
                pass


def test_non_streaming_custom_modules():
    model = AudioSealWM(torch.nn.Identity(), torch.nn.Identity())
    audio = torch.randn(1, 1, 321)
    torch.testing.assert_close(model(audio), 2 * audio)
    with pytest.raises(NotImplementedError, match="Streaming not supported"):
        with model.streaming(1):
            pass


@torch.no_grad()
def test_streaming_cleanup_and_reset(streaming_generator):
    model = streaming_generator
    audio = torch.randn(1, 1, 640)
    message = torch.ones(1, 16, dtype=torch.int32)
    with model.streaming(1):
        expected = model(audio, message=message)
        model.encoder.reset_streaming()
        model.decoder.reset_streaming()
        torch.testing.assert_close(
            model(audio, message=message), expected, atol=0, rtol=0
        )
    with pytest.raises(ValueError, match="test exception"):
        with model.streaming(1):
            model(audio, message=message)
            raise ValueError("test exception")
    assert all(
        not module.is_streaming()
        for module in model.modules()
        if isinstance(module, StreamingModule)
    )
    # Decoder entry can fail after the encoder has already entered streaming.
    model.decoder.model[0].causal = False
    with pytest.raises(AssertionError, match="causal"):
        with model.streaming(1):
            pass
    assert all(
        not module.is_streaming()
        for module in model.modules()
        if isinstance(module, StreamingModule)
    )


@torch.no_grad()
def test_streaming_containers_jit(streaming_generator, tmp_path):
    model = streaming_generator
    audio = torch.randn(1, 1, 640)
    message = torch.ones(1, 16, dtype=torch.int32)
    expected = model(audio, message=message)
    scripted = torch.jit.script(model)
    scripted.save(str(tmp_path / "streaming.jit"))
    scripted = torch.jit.load(str(tmp_path / "streaming.jit"))
    scripted.encoder._start_streaming(1)
    scripted.decoder._start_streaming(1)
    with pytest.raises(torch.jit.Error, match="positive multiple"):
        scripted(audio[..., :321], message=message)
    actual = torch.cat(
        [scripted(chunk, message=message) for chunk in audio.split(320, dim=-1)], dim=-1
    )
    scripted.encoder._stop_streaming()
    scripted.decoder._stop_streaming()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
