"""
Exports the weights in a format that Rust can work with (.npz with everything stripped out
besides weights). 
"""
from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import sys

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
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide a checkpoint file")
    checkpoint = torch.load(sys.argv[1], map_location=torch.device("cpu"))

    new_dict = clean_checkpoint_data_parallel(checkpoint)

    # TODO: These keys will later be non-hardcoded
    keys_to_export = {
        'analysis_transform.conv0.weight',
        'analysis_transform.conv1.weight',
        'analysis_transform.conv2.weight',
        'analysis_transform.conv3.weight',
        'synthesis_transform.conv_transpose0.weight',
        'synthesis_transform.conv_transpose1.weight',
        'synthesis_transform.conv_transpose2.weight',
        'synthesis_transform.conv_transpose3.weight',
        'entropy_bottleneck._offset',
        'entropy_bottleneck._quantized_cdf',
        'entropy_bottleneck._cdf_length'
    }

    exported_dict = { key: new_dict[key] for key in keys_to_export }
    np.savez("weights.npz", **exported_dict)
