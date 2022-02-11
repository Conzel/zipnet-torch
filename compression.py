import numpy as np
import constriction


def cdf_to_pmf(quantized_cdf: np.ndarray, precision_bits: int = 16) -> np.ndarray:
    """Convert quantized CDF to unnormalized PMF.
    """
    assert quantized_cdf.ndim == 1
    assert issubclass(quantized_cdf.dtype.type, np.integer)
    assert precision_bits > 0
    assert precision_bits <= 64
    assert 2**precision_bits == quantized_cdf.max()

    precision_scaling = 1/(2**precision_bits)

    pmf = np.zeros(quantized_cdf.shape[0] - 1, dtype=np.float64)
    for i in range(0, quantized_cdf.shape[0] - 1):
        pmf[i] = (quantized_cdf[i + 1] - quantized_cdf[i]) * precision_scaling

    assert np.allclose(pmf.sum(), 1.0)
    return pmf


def compress_symbols(symbols: np.ndarray, cdf: np.ndarray, cdf_lengths: np.ndarray, precision: int) -> list[int]:
    """
    Compresses the passed symbols with the constriction library.
    If you have a tensor y and want to turn it into symbols, 
    use make_sybols for that purpose.

    CDF is a quantized CDF as the entropy bottlenecks produce it. 
    This means it looks something like this:

    [0, 18, 289, ..., 2**precision]

    Where precision is the precision used to quantize the CDF (usually 16).
    """
    assert symbols.shape[0] == cdf.shape[0]
    num_channels = symbols.shape[0]
    coder = constriction.stream.queue.RangeEncoder()

    for c in range(num_channels):
        pmf = cdf_to_pmf(cdf[c, 0:cdf_lengths[c]], precision).squeeze()

        model = constriction.stream.model.Categorical(pmf)

        coder.encode(symbols[c, :, :].ravel(), model)
    return coder.get_compressed()


def make_symbols(y: np.ndarray, offset: int, symbol_max_per_channel: np.ndarray, precision: int):
    """
    Makes symbols out of a tensor y. 
    This involves quantization and an eventual shift so the
    symbols are natural numbers.
    """
    assert issubclass(y.dtype.type, np.floating)
    num_channels = y.shape[0]
    symbols = np.zeros_like(y, dtype=np.int32)
    for c in range(num_channels):
        quant_range = (offset, offset + symbol_max_per_channel[c] - 1)
        quantized_channel = quantize(y[c, :, :], quant_range) - offset
        assert quantized_channel.max() < symbol_max_per_channel[c]
        assert quantized_channel.min() >= 0

        symbols[c, :, :] = quantized_channel
    return symbols


def quantize(y: np.ndarray, quant_range: tuple[int, int]) -> np.ndarray:
    """
    Quantizes a flat array of values to y to integers. 
    """
    q_min, q_max = quant_range
    assert q_min < q_max
    # TODO: subtract median before this
    y_quantized = y.round().clip(q_min, q_max).astype(np.int32)
    return y_quantized


def decompress_symbols(compressed, target_shape: tuple, cdf: np.ndarray, cdf_lengths: np.ndarray, precision: int) -> np.ndarray:
    """
    Reverse of compress symbols. See there for more information.
    """
    num_channels = target_shape[0]
    size_per_channel = target_shape[1] * target_shape[2]
    coder = constriction.stream.queue.RangeDecoder(compressed)
    symbols = np.zeros(target_shape)
    for c in range(num_channels):
        pmf = cdf_to_pmf(cdf[c, 0:cdf_lengths[c]], precision).squeeze()

        model = constriction.stream.model.Categorical(pmf)

        channel_symbols = coder.decode(model, size_per_channel)

        assert channel_symbols.max() < cdf_lengths[c]
        assert channel_symbols.min() >= 0

        channel_symbols = channel_symbols.reshape(
            target_shape[1], target_shape[2])

        symbols[c, :, :] = channel_symbols

    return symbols
