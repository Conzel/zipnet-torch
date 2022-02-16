import numpy as np
import constriction


def cdf_to_pmf(quantized_cdf: np.ndarray, precision_bits: int = 16) -> np.ndarray:
    """
    Convert the given quantized CDF to unnormalized PMF.
    CDF is quantized in a way such that it starts with 0 and ends with 2**precision_bits.

    PMF is output as floating point numpy array. 
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


def make_symbols(y: np.ndarray, offsets: np.ndarray, symbol_max_per_channel: np.ndarray, means: np.ndarray = None):
    """
    Makes symbols out of a tensor y. 
    This involves quantization in form of rounding and an eventual shift so the
    symbols are natural numbers.

    We make the symbols as follows:

    for each channel:
        1. Subtract the mean of the channel to centralize the distribution (optional)
        2. Build the quantization range, which goes from the given offset (lowest y-value that
        can still be represented) to the highest y-value that can be represented (which is 
        # of elements in the CDF at the given channel + the offset)
        3. Quantize the y-values to the quantization range (they are rounded and clipped to the range)
        4. Subtract the offset so that the symbols are natural numbers
    """
    assert issubclass(y.dtype.type, np.floating)
    assert issubclass(offsets.dtype.type, np.integer)
    num_channels = y.shape[0]
    symbols = np.zeros_like(y, dtype=np.int32)
    for c in range(num_channels):
        if means is not None:
            y_channel = y[c, :, :] - means[c]
        else:
            y_channel = y[c, :, :]
        offset = offsets[c]
        quant_range = (offset, offset + symbol_max_per_channel[c] - 1)
        quantized_channel = quantize(y_channel, quant_range) - offset
        assert quantized_channel.max() < symbol_max_per_channel[c]
        assert quantized_channel.min() >= 0

        symbols[c, :, :] = quantized_channel
    return symbols


def unmake_symbols(symbols: np.ndarray, offsets: np.ndarray, means: np.ndarray = None) -> np.ndarray:
    """
    Takes an array of symbols and restores the original y-values (up to errors
    introduced by quantization).

    This proceeds in reverse order as make_symbols.

    1. Add the offset back in to get values over the original quantization range
    2. Add the mean back in (optional)
    """
    assert issubclass(symbols.dtype.type, np.integer)
    assert issubclass(offsets.dtype.type, np.integer)
    assert symbols.shape[0] == offsets.shape[0]
    # broadcasting ensures everything is fine
    if means is not None:
        assert means.shape[0] == offsets.shape[0]
        return symbols + offsets[:, None, None] + means[:, None, None]
    else:
        return symbols + offsets[:, None, None]


def quantize(y: np.ndarray, quant_range: tuple[int, int]) -> np.ndarray:
    """
    Quantizes an array of values to y to integers. 

    The quantization process is as follows:

    1. Round the values to the nearest integer
    2. Clip the values to the quantization range
    """
    assert y.shape[0]
    q_min, q_max = quant_range
    assert q_min < q_max
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
