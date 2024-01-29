# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import torch


def convert(checkpoint: Path, outdir: Path):
    """Convert the checkpoint to generator and detector"""
    ckpt = torch.load(checkpoint)
    generator_ckpt = {"xp.cfg": ckpt["xp.cfg"], "model": {}}
    detector_ckpt = {"xp.cfg": ckpt["xp.cfg"], "model": {}}

    for layer in ckpt["model"].keys():
        if layer.startswith("detector"):
            detector_ckpt["model"][layer] = ckpt["model"][layer]
        elif layer == "msg_processor.msg_processor.0.weight":
            generator_ckpt["model"]["msg_processor.msg_processor.weight"] = ckpt[
                "model"
            ][layer]
        else:
            generator_ckpt["model"][layer] = ckpt["model"][layer]

    torch.save(generator_ckpt, outdir / (checkpoint.stem + "_generator.pth"))
    torch.save(detector_ckpt, outdir / (checkpoint.stem + "_detector.pth"))


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(convert)
