# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import torch


def convert(checkpoint: Path, outdir: Path, suffix: str = "base"):
    """Convert the checkpoint to generator and detector"""
    ckpt = torch.load(checkpoint)

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
            detector_ckpt["model"][layer] = ckpt["model"][layer]
        elif layer == "msg_processor.msg_processor.0.weight":
            generator_ckpt["model"]["msg_processor.msg_processor.weight"] = ckpt[
                "model"
            ][layer]
        else:
            generator_ckpt["model"][layer] = ckpt["model"][layer]

    torch.save(generator_ckpt, outdir / (checkpoint.stem + f"_generator_{suffix}.pth"))
    torch.save(detector_ckpt, outdir / (checkpoint.stem + f"_detector_{suffix}.pth"))


if __name__ == "__main__":
    import fire

    fire.Fire(convert)
