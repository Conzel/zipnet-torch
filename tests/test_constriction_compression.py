import os
from posixpath import dirname
from compression import compress_symbols, decompress_symbols, encompression_decompression_run, make_symbols, quantize, cdf_to_pmf, unmake_symbols, _mock_quantization
import numpy as np
from compressai._CXX import pmf_to_quantized_cdf
import torch
from export_weights import clean_checkpoint_data_parallel
from models import FactorizedPrior
from PIL import Image
from utils import pil_to_tensor


def test_quantization():
    np.random.seed(5)

    y = np.random.normal(0, 1, (2, 3, 3))
    y_quant = quantize(y, (-2, 2))
    assert np.all(y_quant == np.array(
        [[[0, 0, 2], [0, 0, 2], [-1, -1, 0]], [[0, -1, 0], [0, 1, -2], [-1, 1, 2]]]))


def test_cdf_to_pmf():
    pmf = np.array([[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.25, 0.3, 0.25, 0.1]])
    cdf1 = np.array(pmf_to_quantized_cdf(pmf[0, :].tolist(), 16))
    cdf2 = np.array(pmf_to_quantized_cdf(pmf[1, :].tolist(), 16))
    pmf_rec_1 = cdf_to_pmf(cdf1, 16)
    pmf_rec_2 = cdf_to_pmf(cdf2, 16)
    assert np.abs(pmf_rec_1 - pmf[0, :]).sum() < 1e-3
    assert np.abs(pmf_rec_2 - pmf[1, :]).sum() < 1e-3


def test_compression():
    np.random.seed(5)
    precision = 16
    offsets = np.array([-2, -2])
    means = np.array([0, 0])

    y = np.random.normal(0, 1, (2, 3, 3))
    pmf = np.array([[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.25, 0.3, 0.25, 0.1]])
    cdf1 = pmf_to_quantized_cdf(pmf[0, :].tolist(), precision)
    cdf2 = pmf_to_quantized_cdf(pmf[1, :].tolist(), precision)
    cdf = np.stack((cdf1, cdf2))
    cdf_lengths = np.array([6, 6])

    symbols = make_symbols(y, offsets, cdf_lengths, means)
    compressed = compress_symbols(symbols, cdf, cdf_lengths, precision)
    # This is essentially a regression test.
    assert np.all(compressed == np.array([2461260253, 1328450681]))


def test_decompression():
    np.random.seed(5)
    precision = 16
    offsets = np.array([-2, -2])
    means = np.array([0, 0])

    y = np.random.normal(0, 1, (2, 3, 3))
    pmf = np.array([[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.25, 0.3, 0.25, 0.1]])
    cdf1 = pmf_to_quantized_cdf(pmf[0, :].tolist(), precision)
    cdf2 = pmf_to_quantized_cdf(pmf[1, :].tolist(), precision)
    cdf = np.stack((cdf1, cdf2))
    cdf_lengths = np.array([6, 6])

    symbols = make_symbols(y, offsets, cdf_lengths, means)
    compressed = compress_symbols(symbols, cdf, cdf_lengths, precision)
    symbols_decompressed = decompress_symbols(
        compressed, symbols.shape, cdf, cdf_lengths, precision)
    # This is essentially a regression test.
    assert np.all(symbols == symbols_decompressed)


def test_make_unmake_symbols_random():
    np.random.seed(5)
    offsets = np.array([-6, -7, -7])
    symbols_per_channel = np.array([13, 14, 15])
    y = np.random.random((3, 12, 15))
    means = np.zeros_like(offsets)

    symbols = make_symbols(y, offsets, symbols_per_channel, means)
    y_restored = unmake_symbols(symbols, offsets, means)
    # we can maximally introduce a 0.5 error due to quantization
    assert (np.abs(y_restored - y) <= 0.5).all()
    # sanity check, would be unrealistic to only introduce a 0.1 error
    assert (np.abs(y_restored - y) < 0.1).any()


def test_symbols_retained_during_compression():
    """
    Tests if offset and means have been correctly added and removed again.

    This is the normal process:
    y -> make_symbols -> compress -> decompress -> unmake symbols -> y_hat

    The mock quantization function provides this:
    y --------------------- mock quantization ---------------------> y_hat 

    These two should be identical
    """
    np.random.seed(5)
    offsets = np.array([-6, -7, -7])
    symbols_per_channel = np.array([13, 14, 15])
    y = np.random.random((3, 12, 15))
    means = np.random.random(offsets.shape)

    symbols = make_symbols(y, offsets, symbols_per_channel, means)
    y_restored = unmake_symbols(symbols, offsets, means)

    y_quant_mock = _mock_quantization(y, offsets, symbols_per_channel, means)
    assert np.allclose(y_restored, y_quant_mock)


def test_performance_against_compressai_implementation():
    checkpoint = torch.load(os.path.join(dirname(__file__), "assets/checkpoint.pth.tar"),
                            map_location=torch.device("cpu"))

    new_dict = clean_checkpoint_data_parallel(checkpoint)

    # Restoring the model
    model = FactorizedPrior(128, 192)
    model.load_state_dict(new_dict)
    model.update()

    with Image.open(os.path.join(dirname(__file__), "assets/test-img-link-small.jpg")) as im:
        x = pil_to_tensor(im.crop((0, 0, 256, 256)))
        # the implementation by CompressAI
        s = model.compress(x)
        bytes_cpai = len(s['strings'][0][0])
        x_hat = model.decompress(s["strings"], s["shape"])["x_hat"]
        print(x.shape, x_hat.shape)
        mse_cpai = (x - x_hat).pow(2).mean().item()

        # our implementation
        medians = model.entropy_bottleneck.quantiles[:, 0, 1].detach().numpy()
        y = model.analysis_transform(x)
        compressed, y_quant = encompression_decompression_run(y.squeeze().detach().numpy(), model.entropy_bottleneck._quantized_cdf.numpy(
        ), model.entropy_bottleneck._offset.numpy(), model.entropy_bottleneck._cdf_length.numpy(), 16, means=medians)

        x_hat_constriction = model.synthesis_transform(
            torch.Tensor(y_quant[None, :, :, :])).clamp_(0, 1)
        bytes_constriction = compressed.size*4  # *32/8, as compress is uint32-array
        mse_constriction = (x - x_hat_constriction).pow(2).mean().item()

    print(f"MSE reached: {mse_constriction}")
    print(f"Bytes used: {bytes_constriction}")
    assert mse_cpai == mse_constriction
    assert np.abs(bytes_cpai - bytes_constriction) < 10
    # Mini-regression test
    assert mse_constriction == 0.0018695825710892677
    assert bytes_constriction == 5144
