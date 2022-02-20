# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
            out["bpp"] = bytes / num_pixels
            x_hat = output
        else:
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
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp

def evaluate(test_img, out_net, criterion, get_bpp, bytes=None):
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr_loss = AverageMeter()
    msssim_loss = AverageMeter()
    with torch.no_grad():
        out_criterion = criterion(out_net, test_img, get_bpp, bytes)
        
        bpp_loss.update(out_criterion["bpp"])
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])
        psnr_loss.update(out_criterion["PSNR"])
        msssim_loss.update(out_criterion["ms-ssim"])
    print(
        f"Average losses:"
        f"\tPSNR: {psnr_loss.avg:.3f} |"
        f"\tMS-SSIM: {msssim_loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp: {bpp_loss.avg:.2f} |"
    )
    return bpp_loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluation script.")
    parser.add_argument(
        "--d", type=str, required=True, help="Test image"
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

def main(argv):
    args = parse_args(argv)

    img_tensor, img_pil = load_single_img(args.d)
    model, criterion = load_model(args.checkpoint, args.lmbda)

    print("Evaluating:", args.d)
    print("-"*50)
    x_hat, bytes_compressed, enc_time, dec_time = run_model(model, img_tensor)
    bpp = evaluate(img_tensor, x_hat, criterion, get_bpp=True, bytes=bytes_compressed)

    x_jpeg, bpp_jpeg = find_closest_bpp(bpp, img_pil, fmt="jpeg") 
    print("Closest BPP for:", bpp_jpeg)
    _ = evaluate(img_tensor, pil_to_tensor(x_jpeg), criterion, get_bpp=False, bytes=bpp_jpeg)

    print("Encode time in secs:", enc_time % 60)
    print("Decoder time in secs:", dec_time % 60)


if __name__ == "__main__":
    main(sys.argv[1:])
