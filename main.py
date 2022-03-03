import argparse
from models import load_model
import torch
from PIL import Image
from compression import encompression_decompression_run
from utils import pil_to_tensor, tensor_to_pil


def main(args):
    model = load_model(args.checkpoint)

    with Image.open(args.image) as im:
        x = pil_to_tensor(im)
        # the implementation by CompressAI
        s = model.compress(x)
        print(f"CompressAI size: {len(s['strings'][0][0])} Byte")
        x_hat = model.decompress(s["strings"], s["shape"])["x_hat"]
        im_hat = tensor_to_pil(x_hat)
        im_hat.show()

        # our implementation
        medians = model.entropy_bottleneck.quantiles[:, 0, 1].detach().numpy()
        y = model.analysis_transform(x)
        compressed, y_quant = encompression_decompression_run(y.squeeze().detach().numpy(), model.entropy_bottleneck._quantized_cdf.numpy(
        ), model.entropy_bottleneck._offset.numpy(), model.entropy_bottleneck._cdf_length.numpy(), 16, means=medians)

        x_hat = model.synthesis_transform(
            torch.Tensor(y_quant[None, :, :, :])).clamp_(0, 1)
        # *32/8, as we return an ndarray of uint32
        print(f"Our size: {compressed.size*4} Byte")
        im_hat = tensor_to_pil(x_hat)
        im_hat.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--checkpoint", type=str, required=True)
    args.add_argument("--image", type=str, required=True)
    main(args.parse_args())
