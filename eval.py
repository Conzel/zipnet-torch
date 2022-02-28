'''
# Single img ; single ckpt
python eval.py --single_img True --d tests/assets/test-img-link-small.jpg --single_ckpt --expname test --save_folder tests/assets/results/

# Batch ; folder of ckpt
python eval.py --d tests/assets/images_folder/ --checkpoint True <ckpt directory>
'''

import argparse
import math
import random
import shutil
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
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

from utils import ImageNetDataset
from compression import encompression_decompression_run
from export_weights import clean_checkpoint_data_parallel

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, get_bpp=True, bytes=None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        if get_bpp:
            out["bpp"] = sum(bytes) / num_pixels
            x_hat = output
        else:
            if isinstance(bytes, list):
                bytes = sum(bytes) / len(bytes)
            out["bpp"] = bytes # its BPP in case of JPEG
            x_hat = output
        mse = self.mse(x_hat, target)
        out["mse_loss"] = mse
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp"]
        out["PSNR"] = -10 * math.log10(mse)
        out["ms-ssim"] = ms_ssim(x_hat, target, data_range=1.0).item()
        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_single_img(path, H=256, W=256):
    with Image.open(path) as im:
        pil_img = im.crop((0, 0, H, W))
        x = pil_to_tensor(pil_img)
    return x, pil_img

def load_batch_img(path, H=256, W=256):
    assert(os.path.exists(path))
    img_files = [os.path.join(path, img) for img in os.listdir(path)]
    if len(img_files) < 1:
        print("No images found in directory:", path)
        raise SystemExit(1)

    pil_img_list = [Image.open(img).crop((0, 0, H, W)) for img in img_files]
    tensor_list = [pil_to_tensor(pil_img) for pil_img in pil_img_list]
    return torch.cat(tensor_list, axis=0), pil_img_list

'''
To find the closest bpp for JPEG; we follow compressAI's evaluation scheme:
https://github.com/InterDigitalInc/CompressAI/blob/master/examples/CompressAI%20Inference%20Demo.ipynb
'''
def pillow_encode(img, fmt='jpeg', quality=10):
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp

def find_closest_bpp(target, img, fmt='jpeg'):
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        print(mid, bpp)
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp

def bpp_plot_jpeg(quality, img, fmt='jpeg'):
    rec, bpp = pillow_encode(img, fmt=fmt, quality=int(quality))
    return rec, bpp

def evaluate(test_img, out_net, criterion, get_bpp, results, bytes=None):
    loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr_loss = AverageMeter()
    msssim_loss = AverageMeter()
    with torch.no_grad():
        out_criterion = criterion(out_net, test_img, get_bpp, bytes)
        
        bpp = out_criterion["bpp"]
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])
        psnr_loss.update(out_criterion["PSNR"])
        msssim_loss.update(out_criterion["ms-ssim"])

        results["psnr"].append(psnr_loss.avg)
        results["ms-ssim"].append(msssim_loss.avg)
        results["bpp"].append(bpp)

    print(
        f"Average losses:"
        f"\tPSNR: {psnr_loss.avg:.3f} |"
        f"\tMS-SSIM: {msssim_loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp: {bpp:.2f} |"
    )
    return bpp


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument(
        "--single_img", type=bool, required=False, default=False, help="Flag for evaluating single image"
    )
    parser.add_argument(
        "--d", type=str, required=True, help="Test image in case of single image or else a folder of images"
    )
    parser.add_argument(
        "--single_ckpt", type=bool, required=False, default=False, help="Flag for evaluating single image"
    )
    parser.add_argument(
        "--expname", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--save_folder", type=str, required=True, help="folder to dump json files"
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument("--checkpoint", type=str, default="./tests/assets/checkpoint.pth.tar", help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def load_model(checkpoint_path, lmbda):
    model = FactorizedPrior(128, 192)
    if checkpoint_path:  # load from previous checkpoint
        print("Loading", checkpoint_path)
        checkpoint = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))

        new_dict = clean_checkpoint_data_parallel(checkpoint)
        model.load_state_dict(new_dict)
    else:
        print("Checkpoint path does not exist:", checkpoint_path)
    model.update()
    criterion = RateDistortionLoss(lmbda=lmbda)
    return model, criterion

def run_model(model, test_img):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        test_img = test_img.to(device)
        # our implementation
        medians = model.entropy_bottleneck.quantiles[:, 0, 1].detach().numpy()

        start = time.time()
        y = model.analysis_transform(test_img)
        compressed, y_quant = encompression_decompression_run(y.squeeze().detach().numpy(), model.entropy_bottleneck._quantized_cdf.numpy(
        ), model.entropy_bottleneck._offset.numpy(), model.entropy_bottleneck._cdf_length.numpy(), 16, means=medians)
        enc_time = time.time() - start

        start = time.time()
        x_hat_constriction = model.synthesis_transform(
            torch.Tensor(y_quant[None, :, :, :])).clamp_(0, 1)
        dec_time = time.time() - start
    num = 32 / 8 # compressed is uint32
    return x_hat_constriction, compressed.size * num, enc_time, dec_time

def save_json(args, results, jpeg=False):
    json_save_path = os.path.join(args.save_folder, args.expname + ".json")
    output_dict = {}
    if jpeg == True:
        output_dict["name"] = "JPEG"
    else:
        output_dict["name"] = args.expname
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

def main(argv):
    args = parse_args(argv)
    os.makedirs(args.save_folder, exist_ok=True)
    results = {}
    results_jpeg = {}
    init_dict(results)
    init_dict(results_jpeg)

    if args.single_ckpt:
        checkpoint_list = [args.checkpoint]
    else:
        if not os.path.isdir(args.checkpoint):
            print("Provided path:", args.checkpoint, "is not a valid directory of checkpoints")
            raise SystemExit(1)
        checkpoint_list = [os.path.join(args.checkpoint, ckpt) for ckpt in os.listdir(args.checkpoint)]

    for ckpt in checkpoint_list:
        model, criterion = load_model(ckpt, args.lmbda)
        if args.single_img:
            img_tensor, img_pil = load_single_img(args.d)
            x_hat, bytes_compressed, enc_time, dec_time = run_model(model, img_tensor)
            bytes_compressed = [bytes_compressed]
            print("Encode time in secs:", enc_time % 60)
            print("Decoder time in secs:", dec_time % 60)
        else:
            img_tensor, img_pil_list = load_batch_img(args.d)
            x_hat_list = []
            bytes_compressed = []
            for img in img_tensor:
                x_hat_i, bytes_compressed_i, enc_time, dec_time = run_model(model, img.unsqueeze(0))
                x_hat_list.append(x_hat_i)
                bytes_compressed.append(bytes_compressed_i)
            x_hat = torch.cat(x_hat_list, 0)

        print("Evaluating:", args.d)
        print("-"*50)
        bpp = evaluate(img_tensor, x_hat, criterion, get_bpp=True, results=results, bytes=bytes_compressed)

        if args.single_img:
            for i in range(0, 70):
                x_jpeg, bpp_jpeg = bpp_plot_jpeg(i, img_pil, fmt='jpeg')
                _ = evaluate(img_tensor, pil_to_tensor(x_jpeg), criterion, get_bpp=False, results=results_jpeg, bytes=bpp_jpeg)
        else:
            for i in range(0, 70):
                x_jpeg_list = []
                bpp_jpeg = []
                for img_pil in img_pil_list:
                    x_jpeg_i, bpp_jpeg_i = bpp_plot_jpeg(i, img_pil, fmt='jpeg')
                    x_jpeg_list.append(pil_to_tensor(x_jpeg_i))
                    bpp_jpeg.append(bpp_jpeg_i)
                x_jpeg = torch.cat(x_jpeg_list, 0)
                _ = evaluate(img_tensor, x_jpeg, criterion, get_bpp=False, results=results_jpeg, bytes=bpp_jpeg)

    save_json(args, results)
    args.expname = args.expname + "_jpeg"
    save_json(args, results_jpeg, jpeg=True)


if __name__ == "__main__":
    main(sys.argv[1:])
