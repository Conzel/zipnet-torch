import numpy as np
from models import FactorizedPrior
import torch
from PIL import Image
from compression import encompression_decompression_run, quantize
from export_weights import clean_checkpoint_data_parallel
from utils import pil_to_tensor, tensor_to_pil


checkpoint = torch.load("checkpoint_best_loss.pth.tar",
                        map_location=torch.device("cpu"))

new_dict = clean_checkpoint_data_parallel(checkpoint)

# Restoring the model
model = FactorizedPrior(128, 192)
model.load_state_dict(new_dict)
model.update()

with Image.open("/Users/almico/Downloads/botw-wallpaper.jpg") as im:
    x = pil_to_tensor(im)
    # the implementation by CompressAI
    s = model.compress(x)
    print(f"CompressAI size: {len(s['strings'][0][0])} Byte")
    x_hat = model.decompress(s["strings"], s["shape"])["x_hat"]
    im_hat = tensor_to_pil(x_hat)
    # im_hat.show()

    # our implementation
    y = model.analysis_transform(x)
    compressed, y_quant = encompression_decompression_run(y.squeeze().detach().numpy(), model.entropy_bottleneck._quantized_cdf.numpy(
    ), model.entropy_bottleneck._offset.numpy(), model.entropy_bottleneck._cdf_length.numpy(), 16)
    x_hat = model.synthesis_transform(torch.Tensor(y_quant[None, :, :, :]))
    print(f"Our size: {compressed.size*4} Byte") # *32/8, as we return an ndarray of uint32
    im_hat = tensor_to_pil(x_hat)
    im_hat.show()
