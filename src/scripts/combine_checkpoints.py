# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


def combine_checkpoints(
    generator_checkpoint: str, detector_checkpoint: str, output_checkpoint: str
):
    """Combine split generator and detector checkpoints into a single checkpoint that can be further trained."""
    gen_ckpt = torch.load(generator_checkpoint)
    det_ckpt = torch.load(detector_checkpoint)

    combined_ckpt = {
        "xp.cfg": gen_ckpt["xp.cfg"],  # assuming the configs are identical
        "model": {},
    }

    # add generator layers with appropriate prefix
    for layer in gen_ckpt["model"].keys():
        new_layer = f"generator.{layer}"
        combined_ckpt["model"][new_layer] = gen_ckpt["model"][layer]

    # add detector layers with appropriate prefix
    for layer in det_ckpt["model"].keys():
        new_layer = f"detector.{layer}"
        combined_ckpt["model"][new_layer] = det_ckpt["model"][layer]

    # special case for 'msg_processor.msg_processor.weight'
    if "msg_processor.msg_processor.weight" in gen_ckpt["model"]:
        combined_ckpt["model"]["msg_processor.msg_processor.0.weight"] = gen_ckpt[
            "model"
        ]["msg_processor.msg_processor.weight"]

    torch.save(combined_ckpt, output_checkpoint)


if __name__ == "__main__":
    import fire

    fire.Fire(combine_checkpoints)
