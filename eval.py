'''
# Single img ; single ckpt
python eval.py --single_img True --d tests/assets/test-img-link-small.jpg --single_ckpt --expname test --save_folder results/

# Batch ; folder of ckpt
python eval.py --d tests/assets/images_folder/ --checkpoint True <ckpt directory>
'''

from math import log10, sqrt
from nbformat import current_nbformat
import numpy as np
from models import get_model
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
import torch.nn as nn
from pytorch_msssim import ms_ssim

from export_weights import clean_checkpoint_data_parallel


def load_batch_img(path: str, H=256, W=256) -> tuple[list[torch.Tensor], list[Image.Image]]:
    """
    Loads in images at the folder indicated by the passed path.
    """
    assert(os.path.exists(path))
    img_files = [os.path.join(path, img) for img in os.listdir(
        path) if img.endswith(("jpg", "JPG", "jpeg", "JPEG", "png"))]
    if len(img_files) < 1:
        print("No images found in directory:", path)
        raise SystemExit(1)

    pil_img_list = [Image.open(img).crop((0, 0, H, W)) for img in img_files]
    tensor_list = [pil_to_tensor(pil_img) for pil_img in pil_img_list]
    return tensor_list, pil_img_list


'''
To find the closest bpp for JPEG; we follow compressAI's evaluation scheme:
https://github.com/InterDigitalInc/CompressAI/blob/master/examples/CompressAI%20Inference%20Demo.ipynb
'''


def pillow_encode(img: Image.Image, fmt: str = 'jpeg', quality: int = 10) -> tuple[Image.Image, float]:
    """
    Encodes the given image into JPEG with the given quality. 

    Returns the reconstructed image and the bpp. 
    """
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp


def find_closest_bpp(target_bpp: float, img: Image.Image, fmt='jpeg') -> tuple[Image.Image, float]:
    """
    Tries a range of quality parameters for JPEG until it finds a quality that is 
    very close to the target bpp on the given image.

    Returns achieved bpp and the reconstructed image.
    """
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        print(mid, bpp)
        if bpp > target_bpp:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp


# Whatever this function does?
def bpp_plot_jpeg(quality, img, fmt='jpeg'):
    rec, bpp = pillow_encode(img, fmt=fmt, quality=int(quality))
    return rec, bpp


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument(
        "--images", type=str, required=True, help="Path to an image or folder of images"
    )
    parser.add_argument(
        "--save-folder", type=str, required=True, help="folder to dump json files"
    )
    parser.add_argument("--checkpoints", type=str,
                        default="./tests/assets/checkpoint.pth.tar", help="Path to a checkpoint or folder of checkpoints")
    args = parser.parse_args(argv)
    return args


def load_model(checkpoint_path: str) -> FactorizedPrior:
    """
    Returns the model that is parsed from the given checkpoint path.
    """
    model, _ = parse_checkpoint_to_model(checkpoint_path)
    if checkpoint_path:  # load from previous checkpoint
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))

        new_dict = clean_checkpoint_data_parallel(checkpoint)
        model.load_state_dict(new_dict)
    else:
        raise ValueError("Checkpoint path does not exist:", checkpoint_path)
    model.update(force=True)
    return model


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


def save_json(args, results, name):
    json_save_path = os.path.join(args.save_folder, name + ".json")
    output_dict = {}
    output_dict["name"] = name
    output_dict["results"] = results
    json_data = json.dumps(output_dict)
    jsonFile = open(json_save_path, "w")
    jsonFile.write(json_data)
    jsonFile.close()
    print("saved json at:", json_save_path)


def init_dict(results_dict):
    results_dict["psnr"] = []
    results_dict["ms-ssim"] = []
    results_dict["bpp"] = []


def checkpoint_to_model_name(checkpoint_filename: str):
    model_name_capture = re.search(r".*model=(\w+)-.*", checkpoint_filename)
    if model_name_capture is None or len(model_name_capture.groups()) != 1:
        raise ValueError(f"Invalid checkpoint name: {checkpoint_filename}")
    else:
        return model_name_capture[1]


def parse_checkpoint_to_model(checkpoint_filename: str) -> tuple[FactorizedPrior, float]:
    """
    Parses the name of the checkpoint file and returns the model that was used to
    generate the checkpoint.
    """
    model_name = checkpoint_to_model_name(checkpoint_filename)
    lambda_capture = re.search(r".*lambda=([\.\d]+)-.*", checkpoint_filename)
    if lambda_capture is None or len(lambda_capture.groups()) != 1:
        raise ValueError(f"Invalid checkpoint name: {checkpoint_filename}")
    else:
        lambd = lambda_capture[1]
    return get_model(model_name), float(lambd)


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


def evaluate_jpeg(quality, images) -> tuple[float, float, float]:
    psnr_list = []
    ms_ssim_list = []
    bpp_list = []
    for img_pil in images:
        x_jpeg, bpp_jpeg = bpp_plot_jpeg(quality, img_pil, fmt='jpeg')
        img = pil_to_tensor(img_pil)
        x_jpeg_torch = pil_to_tensor(x_jpeg)
        psnr_list.append(psnr(img.numpy().squeeze(),
                              x_jpeg_torch.numpy().squeeze()))
        ms_ssim_list.append(ms_ssim(img, x_jpeg_torch))
        bpp_list.append(bpp_jpeg)
    return np.mean(psnr_list), np.mean(ms_ssim_list), np.mean(bpp_list)


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
    results_jpeg = {}
    init_dict(results)
    init_dict(results_jpeg)

    if not os.path.isdir(args.checkpoints):
        checkpoint_list = [args.checkpoint]
    else:
        if not os.path.isdir(args.checkpoints):
            print("Provided path:", args.checkpoints,
                  "is not a valid directory of checkpoints")
            raise SystemExit(1)
        checkpoint_list = [os.path.join(args.checkpoints, ckpt)
                           for ckpt in os.listdir(args.checkpoints) if ckpt.endswith(".pth.tar")]
    torch_images, pil_images = load_batch_img(args.images)

    model_name = None
    # evaluating our method
    for checkpoint in checkpoint_list:
        current_model_name = checkpoint_to_model_name(checkpoint)
        if model_name is None:
            model_name = current_model_name
        else:
            if model_name != current_model_name:
                raise ValueError(
                    "Tried to mix evaluation of two different architectures.")
        psnr, ms_ssim, bpp = evaluate_checkpoint(checkpoint, torch_images)
        update_results(results, psnr, ms_ssim, bpp)

    for qual in range(0, 70):
        psnr, ms_ssim, bpp = evaluate_jpeg(qual, pil_images)
        update_results(results_jpeg, psnr, ms_ssim, bpp)

    save_json(args, results, name=model_name)
    save_json(args, results_jpeg, name="jpeg")


if __name__ == "__main__":
    main(sys.argv[1:])
