# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import torch


def convert(checkpoint: Path, outdir: Path, prefix: str = "", suffix: str = "base"):
    """Convert the checkpoint to generator and detector"""
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")

    # keep inference-related params only
    infer_cfg = {
        "seanet": ckpt["xp.cfg"]["seanet"],
        "channels": ckpt["xp.cfg"]["channels"],
        "dtype": ckpt["xp.cfg"]["dtype"],
        "sample_rate": ckpt["xp.cfg"]["sample_rate"],
    }

    generator_ckpt = {"xp.cfg": infer_cfg, "model": {}}
    detector_ckpt = {"xp.cfg": infer_cfg, "model": {}}

    for layer in ckpt["model"].keys():
        if layer.startswith("detector"):
            new_layer = layer[len("detector.") :]  # remove "detector." prefix
            detector_ckpt["model"][new_layer] = ckpt["model"][layer]  # type: ignore
        elif layer == "msg_processor.msg_processor.0.weight":
            generator_ckpt["model"]["msg_processor.msg_processor.weight"] = ckpt[  # type: ignore
                "model"
            ][layer]
        else:
            assert layer.startswith("generator."), f"Unknown layer: {layer}"
            new_layer = layer[len("generator.") :]
            generator_ckpt["model"][new_layer] = ckpt["model"][layer]  # type: ignore

    torch.save(generator_ckpt, Path(outdir) / (prefix + f"generator_{suffix}.pth"))
    torch.save(detector_ckpt, Path(outdir) / (prefix + f"detector_{suffix}.pth"))


if __name__ == "__main__":
    import fire

    fire.Fire(convert)
