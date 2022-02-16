from compression import compress_symbols, decompress_symbols, make_symbols, quantize, cdf_to_pmf, unmake_symbols
import numpy as np
from compressai._CXX import pmf_to_quantized_cdf


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

    y = np.random.normal(0, 1, (2, 3, 3))
    pmf = np.array([[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.25, 0.3, 0.25, 0.1]])
    cdf1 = pmf_to_quantized_cdf(pmf[0, :].tolist(), precision)
    cdf2 = pmf_to_quantized_cdf(pmf[1, :].tolist(), precision)
    cdf = np.stack((cdf1, cdf2))
    cdf_lengths = np.array([6, 6])

    symbols = make_symbols(y, offsets, cdf_lengths)
    compressed = compress_symbols(symbols, cdf, cdf_lengths, precision)
    # This is essentially a regression test.
    assert np.all(compressed == np.array([2461260253, 1328450681]))


def test_decompression():
    np.random.seed(5)
    precision = 16
    offsets = np.array([-2, -2])

    y = np.random.normal(0, 1, (2, 3, 3))
    pmf = np.array([[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.25, 0.3, 0.25, 0.1]])
    cdf1 = pmf_to_quantized_cdf(pmf[0, :].tolist(), precision)
    cdf2 = pmf_to_quantized_cdf(pmf[1, :].tolist(), precision)
    cdf = np.stack((cdf1, cdf2))
    cdf_lengths = np.array([6, 6])

    symbols = make_symbols(y, offsets, cdf_lengths)
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

    symbols = make_symbols(
        y, offsets, symbol_max_per_channel=symbols_per_channel)
    y_restored = unmake_symbols(symbols, offsets)
    # we can maximally introduce a 0.5 error due to quantization
    assert (np.abs(y_restored - y) <= 0.5).all()
    # sanity check, would be unrealistic to only introduce a 0.1 error
    assert (np.abs(y_restored -y) < 0.1).any()
