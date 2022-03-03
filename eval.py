"""
Produces R/D results for plotting.
Usage:
python3 eval.py --checkpoints path/to/checkpoints --images /path/to/images --save-folder results

For further info:
python3 eval.py --help
"""

from math import log10
import numpy as np
from models import load_model
import argparse
import re
import sys
from models import FactorizedPrior
from PIL import Image
from utils import pil_to_tensor
import io
import time
import os
import json

import torch
from pytorch_msssim import ms_ssim

from export_weights import clean_checkpoint_data_parallel


def load_batch_img(path: str, H=256, W=256) -> tuple[list[torch.Tensor], list[Image.Image]]:
    """
    Loads in images at the folder indicated by the passed path.
    """
    assert(os.path.exists(path))
    if not os.path.isdir(path):
        img_files = [path]
    else:
        img_files = [os.path.join(path, img) for img in os.listdir(
            path) if img.endswith(("jpg", "JPG", "jpeg", "JPEG", "png"))]
        if len(img_files) < 1:
            print("No images found in directory:", path)
            raise SystemExit(1)

    pil_img_list = [Image.open(img).crop((0, 0, H, W)) for img in img_files]
    tensor_list = [pil_to_tensor(pil_img) for pil_img in pil_img_list]
    return tensor_list, pil_img_list


def pillow_encode(img: Image.Image, fmt: str = 'jpeg', quality: int = 10) -> tuple[Image.Image, float]:
    """
    Encodes the given image into the specified format (usually JPEG) with the given quality. 

    Returns the reconstructed image and the bpp. 
    """
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Evaluation script for R/D curves.")
    parser.add_argument(
        "--images", type=str, required=True, help="Path to an image or folder of images"
    )
    parser.add_argument(
        "--save-folder", type=str, required=True, help="Folder to dump json files"
    )
    parser.add_argument("--checkpoints", type=str,
                        help="Path to a checkpoint or folder of checkpoints", required=False)
    parser.add_argument("--baseline", type=str,
                        choices=["jpeg"], required=False, help="Name of the baseline to use.")
    args = parser.parse_args(argv)
    return args


def run_compression(model: FactorizedPrior, test_img: torch.Tensor) -> tuple[torch.Tensor, int, float, float]:
    """
    Runs the given model on the image. Uses constriction to do compression and
    actually performs it (instead of using some intermediate measure of bpp like
    entropy).

    Return the reconstructed image, the size of the compressed representation in bits, 
    the encoding time and the decoding time.
    """
    with torch.no_grad():
        start = time.time()
        compressed, y_hat = model.compress_constriction(test_img)
        enc_time = time.time() - start

        start = time.time()
        x_hat_constriction = model.decompress_constriction(
            compressed, y_hat.shape)
        dec_time = time.time() - start
    return x_hat_constriction, compressed.size * 32, enc_time, dec_time


def save_json(save_folder: str, results: dict[str, list[float]], name: str):
    """
    Saves results in a .json file that is called "name"
    under the specified folder.
    """
    results = sort_results(results)
    json_save_path = os.path.join(save_folder, name + ".json")
    output_dict = {}
    output_dict["name"] = name
    output_dict["results"] = results
    json_data = json.dumps(output_dict)
    jsonFile = open(json_save_path, "w")
    jsonFile.write(json_data)
    jsonFile.close()
    print("saved json at:", json_save_path)


def init_dict(results_dict: dict):
    results_dict["psnr"] = []
    results_dict["ms-ssim"] = []
    results_dict["bpp"] = []


def evaluate_checkpoint(checkpoint: str, image_list: list[torch.Tensor]) -> tuple[float, float, float]:
    """
    Evaluates the checkpoint at the given path on the given image list.
    Returns the average PSNR, average MS-SSIM and average bpp in that order.
    """
    print("Evaluating model at checkpoint:", checkpoint)
    print("-"*50)

    model = load_model(checkpoint)

    psnr_list = []
    ms_ssim_list = []
    bpp_list = []
    for img in image_list:
        x_hat_i, bytes_compressed_i, _, _ = run_compression(model, img)
        psnr_list.append(psnr(img.numpy().squeeze(),
                         x_hat_i.numpy().squeeze()))
        ms_ssim_list.append(ms_ssim(img, x_hat_i))
        num_pixel = img.shape[2] * img.shape[3]
        bpp_list.append(bytes_compressed_i / num_pixel)
    return np.mean(psnr_list), np.mean(ms_ssim_list), np.mean(bpp_list)


def psnr(orig: np.ndarray, rec: np.ndarray) -> float:
    """
    From https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio.
    """
    mse = np.mean((orig - rec) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        raise ValueError("Images passed are identical")
    # as our tensors are scaled to 1.0, we need to set
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel) - 10 * log10(mse)
    return psnr


def evaluate_baseline(quality: float, images: list[Image.Image], fmt: str) -> tuple[float, float, float]:
    psnr_list = []
    ms_ssim_list = []
    bpp_list = []
    for img_pil in images:
        x_baseline, bpp_baseline = pillow_encode(
            img_pil, fmt=fmt, quality=int(quality))
        img = pil_to_tensor(img_pil)
        x_baseline_torch = pil_to_tensor(x_baseline)
        psnr_list.append(psnr(img.numpy().squeeze(),
                              x_baseline_torch.numpy().squeeze()))
        ms_ssim_list.append(ms_ssim(img, x_baseline_torch))
        bpp_list.append(bpp_baseline)
    return np.mean(psnr_list), np.mean(ms_ssim_list), np.mean(bpp_list)


def sort_results(results_dict: dict[str, list[float]]):
    """
    Sorts the resulting dictionary. Returns sorted dictionary.
    """
    sorted_dict = {}
    order = np.argsort(results_dict["bpp"])
    for key, val in results_dict.items():
        sorted_dict[key] = [val[i] for i in order]
    return sorted_dict


def update_results(results_dict: dict[str, list[float]], psnr: float, ms_ssim: float, bpp: float):
    """
    Adds psnr, msessim and bpp to the results dict.

    Modifies the results_dict in-place.
    """
    # Need to cast to float as we might return numpy floats.
    results_dict["psnr"].append(float(psnr))
    results_dict["ms-ssim"].append(float(ms_ssim))
    results_dict["bpp"].append(float(bpp))


def main(argv):
    args = parse_args(argv)
    os.makedirs(args.save_folder, exist_ok=True)
    results = {}
    results_baseline = {}
    init_dict(results)
    init_dict(results_baseline)

    torch_images, pil_images = load_batch_img(args.images)

    if args.checkpoints is not None:
        if not os.path.isdir(args.checkpoints):
            checkpoint_list = [args.checkpoints]
        else:
            checkpoint_list = [os.path.join(args.checkpoints, ckpt)
                               for ckpt in os.listdir(args.checkpoints) if ckpt.endswith(".pth.tar")]

        model_names = set()
        # evaluating our method
        for checkpoint in checkpoint_list:
            current_model_name = checkpoint_to_model_name(checkpoint)
            model_names.add(current_model_name)
            if len(model_names) > 1:
                raise ValueError(
                    f"Tried to mix evaluation of two different architectures: {model_names}.")
            psnr, ms_ssim, bpp = evaluate_checkpoint(checkpoint, torch_images)
            update_results(results, psnr, ms_ssim, bpp)

        save_json(args.save_folder, results, name=list(model_names)[0])

    if args.baseline is not None:
        print(f"Evaluating baseline {args.baseline}...")
        for qual in range(0, 95):
            psnr, ms_ssim, bpp = evaluate_baseline(
                qual, pil_images, args.baseline)
            update_results(results_baseline, psnr, ms_ssim, bpp)

        save_json(args.save_folder, results_baseline, name=args.baseline)

    if args.baseline is None and args.checkpoints is None:
        print("Warning: Neither baseline nor checkpoints specified. This script exited without doing anything.")


if __name__ == "__main__":
    main(sys.argv[1:])
