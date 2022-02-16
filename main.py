from models import FactorizedPrior
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from typing import Dict
from compression import encompression_decompression_run


def tensor_to_pil(x: torch.Tensor) -> Image:
    """Convert a tensor to a PIL image."""
    return ToPILImage()(x.squeeze())


def pil_to_tensor(im: Image) -> torch.Tensor:
    """Convert a PIL image to a tensor."""
    tensor = ToTensor()(im)
    if tensor.shape[0] == 3:
        return tensor[None, :, :, :]
    else:
        return tensor[None, :3, :, :]


def clean_checkpoint_data_parallel(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    # If the model was trained with DataParallel, we will have to remove
    # the .module prefix from the keys.
    # See:
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/8
    new_dict = {}
    for key in checkpoint["state_dict"]:
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        new_dict[new_key] = checkpoint["state_dict"][key]
    return new_dict


checkpoint = torch.load("checkpoint_best_loss.pth.tar",
                        map_location=torch.device("cpu"))

new_dict = clean_checkpoint_data_parallel(checkpoint)

# Restoring the model
model = FactorizedPrior(128, 192)
model.load_state_dict(new_dict)
model.update()

# with Image.open("/Users/almico/Downloads/link.png") as im:
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
