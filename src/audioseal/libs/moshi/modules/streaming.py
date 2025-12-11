# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Streaming module API that should be implemented by all Streaming components,
"""

import itertools
import math
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


class Resetable(tp.Protocol):
    def init_stream(self) -> None:
        pass

    def is_streamable(self) -> bool: ...

    def reset(self) -> None:
        pass


State = tp.TypeVar("State", bound=Resetable)


@torch.jit.script
@dataclass
class _NullState:
    streamable: bool = False

    def init_stream(self) -> None:
        self.streamable = True

    def is_streamable(self) -> bool:
        return self.streamable

    def reset(self) -> None:
        pass


class StreamingModule(nn.Module, tp.Generic[State]):
    """Common API for streaming components.

    Each streaming component has a streaming state, which is just a dict[str, Tensor].
    By convention, the first dim of each tensor must be the batch size.
    Don't use dots in the key names, as this would clash with submodules
    (like in state_dict).

    If `self._is_streaming()` is True, the component should use and remember
    the proper state inside `self._streaming_state`.

    To set a streaming component in streaming state, use

        with module.streaming():
            ...

    This will automatically reset the streaming state when exiting the context manager.
    This also automatically propagates to all streaming children module.

    Some module might also implement the `StreamingModule.flush` method, although
    this one is trickier, as all parents module must be StreamingModule and implement
    it as well for it to work properly. See `StreamingSequential` after.
    """

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State = self._empty_state()
        self._streaming_propagate: bool = True

    @torch.jit.export
    def is_streaming(self):
        return self._streaming_state.is_streamable()

    def set_streaming_propagate(self, streaming_propagate: bool):
        self._streaming_propagate = streaming_propagate

    def _empty_state(self) -> State:
        raise NotImplementedError

    @torch.jit.export
    def _start_streaming(self, batch_size: int):
        self._streaming_state = self._init_streaming_state(batch_size)
        self._streaming_state.init_stream()
        if self._streaming_propagate:
            for _, child in self.named_children():
                if hasattr(child, "_start_streaming"):
                    child._start_streaming(batch_size)  # type: ignore

    @torch.jit.export
    def _stop_streaming(self):
        self._streaming_state = self._empty_state()
        if self._streaming_propagate:
            for _, child in self.named_children():
                if hasattr(child, "_stop_streaming"):
                    child._stop_streaming()  # type: ignore

    def _init_streaming_state(self, batch_size: int) -> State:
        raise NotImplementedError

    def streaming_forever(self, batch_size: int):
        self._start_streaming(batch_size)

    @contextmanager
    def streaming(self, batch_size: int):
        """Context manager to enter streaming mode. Reset streaming state on exit."""

        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming()

    @torch.jit.export
    def reset_streaming(self):
        """Reset the streaming state."""

        state = self._streaming_state
        if not state.is_streamable():
            raise ValueError("Trying to reset non-streamable module.")
        state.reset()

        if self._streaming_propagate:
            for _, child in self.named_children():
                if hasattr(child, "reset_streaming"):
                    child.reset_streaming()

    @torch.jit.export
    def _add_streaming_state(self, prefix: str, state: tp.Dict[str, tp.Any]) -> None:
        """Add the streaming state to the given state dict."""
        state[prefix] = self._streaming_state
        if self._streaming_propagate:
            for name, module in self.named_children():
                if hasattr(module, "_add_streaming_state"):
                    name = prefix + "." + name
                    module._add_streaming_state(name, state)  # type: ignore

    @torch.jit.export
    def get_streaming_state(self) -> tp.Dict[str, tp.Any]:
        """Return the complete streaming state, including that of sub-modules."""
        state: tp.Dict[str, tp.Any] = {}
        self._add_streaming_state("", state)
        return state

    def _pop_streaming_state(self, prefix: str, state: tp.Dict[str, tp.Any]) -> None:
        """Set the streaming state, including that of sub-modules."""
        if prefix in state:
            self._streaming_state = state[prefix]
            state.pop(prefix)
        else:
            raise RuntimeError(f"Expected to find a streaming state for {prefix}.")
        if self._streaming_propagate:
            for name, module in self.named_children():
                if hasattr(module, "_pop_streaming_state"):
                    name = prefix + "." + name
                    module._pop_streaming_state(name, state)  # type: ignore

    def set_streaming_state(self, state: tp.Dict[str, tp.Any]):
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        self._pop_streaming_state("", state)
        if state:
            raise RuntimeError(f"Some states were not consumed: {list(state.keys())}")


class StreamingContainer(StreamingModule[_NullState]):
    @torch.jit.export
    def _init_streaming_state(self, batch_size: int) -> _NullState:
        return _NullState()

    @torch.jit.export
    def _empty_state(self) -> _NullState:
        return _NullState()


@torch.jit.script
@dataclass
class _StreamingAddState:
    previous_x: tp.Optional[torch.Tensor] = None
    previous_y: tp.Optional[torch.Tensor] = None
    streamable: bool = False

    def init_stream(self) -> None:
        self.streamable = True

    def is_streamable(self) -> bool:
        return self.streamable

    def reset(self):
        self.previous_x = None
        self.previous_y = None


class StreamingAdd(StreamingModule[_StreamingAddState]):
    @torch.jit.export
    def _init_streaming_state(self, batch_size: int) -> _StreamingAddState:
        return _StreamingAddState()

    def _empty_state(self) -> _StreamingAddState:
        return _StreamingAddState()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if not self._streaming_state.is_streamable():
            return x + y
        else:
            prev_x = self._streaming_state.previous_x
            prev_y = self._streaming_state.previous_y
            if prev_x is not None:
                x = torch.cat([prev_x, x], dim=-1)
            if prev_y is not None:
                y = torch.cat([prev_y, y], dim=-1)
            m_l = min(x.shape[-1], y.shape[-1])
            self._streaming_state.previous_x = x[..., m_l:]
            self._streaming_state.previous_y = y[..., m_l:]
            return x[..., :m_l] + y[..., :m_l]


@torch.jit.script
@dataclass
class _StreamingConvState:
    previous: tp.Optional[torch.Tensor] = None
    streamable: bool = False

    def init_stream(self) -> None:
        self.streamable = True

    def is_streamable(self) -> bool:
        return self.streamable

    def reset(self):
        self.previous = None


class RawStreamingConv1d(StreamingModule[_StreamingConvState]):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inner_conv = nn.Conv1d(*args, **kwargs)
        assert self.inner_conv.padding[0] == 0, "Padding should be handled outside."
        assert (
            self.inner_conv.stride[0] <= self.inner_conv.kernel_size[0]
        ), "stride must be less than kernel_size."

    @torch.jit.export
    def _init_streaming_state(self, batch_size: int) -> _StreamingConvState:
        return _StreamingConvState()

    @torch.jit.export
    def _empty_state(self) -> _StreamingConvState:
        return _StreamingConvState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.inner_conv.stride[0]
        # Effective kernel size accounting for dilation.
        kernel = (self.inner_conv.kernel_size[0] - 1) * self.inner_conv.dilation[0] + 1
        if not self._streaming_state.is_streamable():
            return self.inner_conv.forward(input)
        else:
            # Due to the potential overlap, we might have some cache of the previous time steps.
            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # We now compute the number of full convolution frames, i.e. the frames
            # that are ready to be computed.
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            offset = num_frames * stride
            # We will compute `num_frames` outputs, and we are advancing by `stride`
            # for each of the frame, so we know the data before `stride * num_frames`
            # will never be used again.
            self._streaming_state.previous = input[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                out = self.inner_conv.forward(input[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B,
                    self.inner_conv.out_channels,
                    0,
                    device=input.device,
                    dtype=input.dtype,
                )
            return out


@torch.jit.script
@dataclass
class _StreamingConvTrState:
    partial: tp.Optional[torch.Tensor] = None
    streamable: bool = False

    def init_stream(self) -> None:
        self.streamable = True

    def is_streamable(self) -> bool:
        return self.streamable

    def reset(self):
        self.partial = None


class RawStreamingConvTranspose1d(StreamingModule[_StreamingConvTrState]):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.inner_conv = nn.ConvTranspose1d(*args, **kwargs)
        assert self.inner_conv.padding[0] == 0, "Padding should be handled outside."
        assert self.inner_conv.dilation[0] == 1, "No dilation for now"
        assert (
            self.inner_conv.stride[0] <= self.inner_conv.kernel_size[0]
        ), "stride must be less than kernel_size."
        assert self.inner_conv.output_padding[0] == 0, "Output padding not supported."

    @torch.jit.export
    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTrState:
        return _StreamingConvTrState()

    @torch.jit.export
    def _empty_state(self) -> _StreamingConvTrState:
        return _StreamingConvTrState()

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        B, C, T = x.shape
        stride = self.inner_conv.stride[0]
        kernel = self.inner_conv.kernel_size[0]
        if not self._streaming_state.is_streamable():
            return self.inner_conv.forward(x)
        else:
            if T == 0:
                return torch.empty(
                    B, self.inner_conv.out_channels, 0, device=x.device, dtype=x.dtype
                )
            out = self.inner_conv.forward(x)
            OT = out.shape[-1]
            assert hasattr(self._streaming_state, "partial")
            partial = self._streaming_state.partial
            if partial is not None:
                # Due to the potential overlap, the rightmost output of the conv transpose is not
                # ready to be output, as it will receive contributions from the next input frames.
                # Here we recover those `partial` output frames. We know that the first time step
                # of the `partial` tensor corresponds to the first time step of `out` as anything
                # coming before the first time step of `out` would have been already flushed.
                PT = partial.shape[-1]
                if self.inner_conv.bias is not None:
                    # type: ignore
                    _conv: tp.Optional[torch.Tensor] = self.inner_conv.bias
                    assert _conv is not None
                    out[..., :PT] += partial - _conv[:, None]
                else:
                    out[..., :PT] += partial
            # The input is T, the output is S * (T - 1) + K.
            # The offset of the left of the next frame will be S * T
            # so everything between 0 and S * T is ready to be output, and we need
            # to keep in the internal state everything beyond that, i.e. S (T - 1) + K - S T = K - S
            invalid_steps = kernel - stride
            partial = out[..., OT - invalid_steps :]
            out = out[..., : OT - invalid_steps]
            self._streaming_state.partial = partial
            return out


def test():
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        # Avoid the cuda optimizations that would take place on single precision
        # floats for convolutions.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    kernel_sizes = [1, 3, 4, 8, 15, 16]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    chin = 6
    chout = 12

    for kernel, stride in itertools.product(kernel_sizes, strides):
        if stride > kernel:
            continue
        conv = RawStreamingConv1d(chin, chout, kernel, stride).to(device)
        convtr = RawStreamingConvTranspose1d(chout, chin, kernel, stride).to(device)

        for length in [4, 8, 32, 54, 65, 128, 1043]:
            print(f"ksize {kernel} strides {stride} len {length}")
            if length < kernel:
                continue
            batch_size = 3
            x = torch.randn(batch_size, chin, length).to(device)
            y = conv(x)
            z = convtr(y)
            for chunk_size in [1, 3, 5, 8]:
                ys = []
                zs = []
                with conv.streaming(batch_size), convtr.streaming(batch_size):
                    for offset in range(0, length, chunk_size):
                        chunk = x[..., offset : offset + chunk_size]
                        ys.append(conv(chunk))
                        zs.append(convtr(ys[-1]))
                y_stream = torch.cat(ys, dim=-1)
                z_stream = torch.cat(zs, dim=-1)
                y = y[..., : y_stream.shape[-1]]
                z = z[..., : z_stream.shape[-1]]
                assert y.shape == y_stream.shape, (y.shape, y_stream.shape)
                delta = (y_stream - y).norm() / y.norm()
                assert delta <= 1e-6, delta
                num_frames = int((length - kernel) / stride) + 1
                assert num_frames == y_stream.shape[-1]

                assert z.shape == z_stream.shape, (z.shape, z_stream.shape)
                delta = (z_stream - z).norm() / z.norm()
                assert delta <= 1e-6, (delta, (z_stream - z).abs().mean(dim=(0, 1)))


if __name__ == "__main__":
    with torch.no_grad():
        test()
