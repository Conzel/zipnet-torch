"""
Exports the weights in a format that Rust can work with (.npz with everything stripped out
besides weights).
"""
from multiprocessing.sharedctypes import Value
import pathlib
import torch
import numpy as np
import sys
import argparse

from models import FactorizedPrior
from training import get_model


def clean_checkpoint_data_parallel(checkpoint: dict) -> dict[str, torch.Tensor]:
    """
    If the model was trained with DataParallel, we will have to remove
    the .module prefix from the keys.

    This function is responsible for that.

    See:
    https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/8
    """
    new_dict = {}
    for key in checkpoint["state_dict"]:
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        new_dict[new_key] = checkpoint["state_dict"][key]
    return new_dict


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    new_dict = clean_checkpoint_data_parallel(checkpoint)

    model = get_model(args.model)
    model.load_state_dict(new_dict)
    model.update()

    # model update populates some important variables,
    # this is why we have to call it here.
    state_dict = model.state_dict()

    # TODO: These keys will later be non-hardcoded
    keys_to_export = {
        'entropy_bottleneck.quantiles',
        'entropy_bottleneck._offset',
        'entropy_bottleneck._quantized_cdf',
        'entropy_bottleneck._cdf_length',
        'analysis_transform.conv0.weight',
        'analysis_transform.gdn0.beta',
        'analysis_transform.gdn0.gamma',
        'analysis_transform.conv1.weight',
        'analysis_transform.gdn2.beta',
        'analysis_transform.gdn2.gamma',
        'analysis_transform.conv2.weight',
        'analysis_transform.gdn3.beta',
        'analysis_transform.gdn3.gamma',
        'analysis_transform.conv3.weight',
        'synthesis_transform.conv_transpose0.weight',
        'synthesis_transform.igdn0.beta',
        'synthesis_transform.igdn0.gamma',
        'synthesis_transform.conv_transpose1.weight',
        'synthesis_transform.igdn1.beta',
        'synthesis_transform.igdn1.gamma',
        'synthesis_transform.conv_transpose2.weight',
        'synthesis_transform.igdn2.beta',
        'synthesis_transform.igdn2.gamma',
        'synthesis_transform.conv_transpose3.weight'
    }

    exported_dict = {key: state_dict[key] for key in keys_to_export}
    exported_dict["entropy_bottleneck._medians"] = state_dict["entropy_bottleneck.quantiles"][:, :, 1:2].squeeze()
    np.savez(args.out, **exported_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", metavar="CHECKPOINT",
                        type=pathlib.Path)
    parser.add_argument("--out", metavar="OUT",
                        type=pathlib.Path, default="weights.npz")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model to train"
    )
    args = parser.parse_args()
    main(args)
