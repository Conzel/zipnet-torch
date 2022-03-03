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

from collections import OrderedDict
from typing import Callable, Optional, OrderedDict
import warnings
import numpy as np
import torch

import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1

from utils import conv, conv_transpose, update_registered_buffers
from compression import decompress_symbols, encompression_decompression_run, unmake_symbols


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, entropy_bottleneck_channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(
            entropy_bottleneck_channels)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        # # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    We use this as a baseclass, the actual models will be defined by subclasses,
    which are responsible for instantiating an analysis and synthesis transform.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)
        self.N = N
        self.M = M
        self.synthesis_transform: Callable
        self.analysis_transform: Callable

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        y = self.analysis_transform(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.synthesis_transform(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def compress(self, x):
        y = self.analysis_transform(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.synthesis_transform(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def compress_decompress_constriction(self, x: torch.Tensor):
        """
        Performs both compression and decompression on an image. 
        Used for debugging and performance analysis.
        """
        medians = self.entropy_bottleneck.quantiles[:, 0, 1].detach().numpy()
        y = self.analysis_transform(x)
        compressed, y_quant = encompression_decompression_run(y.squeeze().detach().numpy(), self.entropy_bottleneck._quantized_cdf.numpy(
        ), self.entropy_bottleneck._offset.numpy(), self.entropy_bottleneck._cdf_length.numpy(), 16, means=medians)
        x_hat_constriction = self.synthesis_transform(
            torch.Tensor(y_quant[None, :, :, :])).clamp_(0, 1)
        return compressed, y_quant, x_hat_constriction

    def compress_constriction(self, x: torch.Tensor):
        """
        Compresses an using constriction.
        """
        medians = self.entropy_bottleneck.quantiles[:, 0, 1].detach().numpy()
        y = self.analysis_transform(x)
        compressed, y_quant = encompression_decompression_run(y.squeeze().detach().numpy(), self.entropy_bottleneck._quantized_cdf.numpy(
        ), self.entropy_bottleneck._offset.numpy(), self.entropy_bottleneck._cdf_length.numpy(), 16, means=medians)
        return compressed, y_quant

    def decompress_constriction(self, compressed_representation: np.ndarray, latent_shape: tuple):
        """
        Decompresses a compressed representation of an image using constriction.
        """
        medians = self.entropy_bottleneck.quantiles[:, 0, 1].detach().numpy()
        offsets = self.entropy_bottleneck._offset.numpy()
        decompressed_symbols = decompress_symbols(
            compressed_representation, latent_shape, self.entropy_bottleneck._quantized_cdf.numpy(), self.entropy_bottleneck._cdf_length.numpy(), 16)
        y_tilde = unmake_symbols(decompressed_symbols, offsets, medians)
        x_hat_constriction = self.synthesis_transform(
            torch.Tensor(y_tilde[None, :, :, :])).clamp_(0, 1)
        return x_hat_constriction


class FactorizedPriorRelu(FactorizedPrior):
    """
    Same as the factorized prior model, but we are using GDN layers instead
    of simple ReLU.
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.analysis_transform = nn.Sequential(
            OrderedDict([
                ("conv0", conv(3, N)),
                ("relu0", nn.ReLU()),
                ("conv1", conv(N, N)),
                ("relu1", nn.ReLU()),
                ("conv2", conv(N, N)),
                ("relu2", nn.ReLU()),
                ("conv3", conv(N, M)),
            ]))

        self.synthesis_transform = nn.Sequential(
            OrderedDict([
                ("conv_transpose0", conv_transpose(M, N)),
                ("relu0", nn.ReLU()),
                ("conv_transpose1", conv_transpose(N, N)),
                ("relu1", nn.ReLU()),
                ("conv_transpose2", conv_transpose(N, N)),
                ("relu2", nn.ReLU()),
                ("conv_transpose3", conv_transpose(N, 3)),
            ]))


class FactorizedPriorGdn(FactorizedPrior):
    """
    Same as the factorized prior model, but we are using GDN layers instead
    of simple ReLU.
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.analysis_transform = nn.Sequential(
            OrderedDict([
                ("conv0", conv(3, N)),
                ("gdn0", GDN1(N)),
                ("conv1", conv(N, N)),
                ("gdn1", GDN1(N)),
                ("conv2", conv(N, N)),
                ("gdn2", GDN1(N)),
                ("conv3", conv(N, M)),
            ]))

        self.synthesis_transform = nn.Sequential(
            OrderedDict([
                ("conv_transpose0", conv_transpose(M, N)),
                ("igdn0", GDN1(N, inverse=True)),
                ("conv_transpose1", conv_transpose(N, N)),
                ("igdn1", GDN1(N, inverse=True)),
                ("conv_transpose2", conv_transpose(N, N)),
                ("igdn2", GDN1(N, inverse=True)),
                ("conv_transpose3", conv_transpose(N, 3)),
            ]))


class FactorizedPriorGdnUpsampling(FactorizedPrior):
    """
    Same as the factorized prior model, but we are using GDN layers instead
    of simple ReLU.
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.analysis_transform = nn.Sequential(
            OrderedDict([
                ("conv0", conv(3, N)),
                ("gdn0", GDN1(N)),
                ("conv1", conv(N, N)),
                ("gdn1", GDN1(N)),
                ("conv2", conv(N, N)),
                ("gdn2", GDN1(N)),
                ("conv3", conv(N, M)),
            ]))

        self.synthesis_transform = nn.Sequential(
            OrderedDict([
                ("upsample0", nn.Upsample(scale_factor=2, mode="nearest")),
                ("convs0", conv(M, N, stride=1)),
                ("igdn0", GDN1(N, inverse=True)),
                ("upsample1", nn.Upsample(scale_factor=2, mode="nearest")),
                ("convs1", conv(N, N, stride=1)),
                ("igdn1", GDN1(N, inverse=True)),
                ("upsample2", nn.Upsample(scale_factor=2, mode="nearest")),
                ("convs2", conv(N, N, stride=1)),
                ("igdn2", GDN1(N, inverse=True)),
                ("upsample3", nn.Upsample(scale_factor=2, mode="nearest")),
                ("convs3", conv(N, 3, stride=1)),
            ]))


class FactorizedPriorGdnUpsamplingBalle(FactorizedPrior):
    """
    Directly taken from the original paper "End-to-End optimized image compression", Balle et al, 2016
    This seems to run a bit slow.
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.analysis_transform = nn.Sequential(
            OrderedDict([
                ("conv0", conv(3, N, kernel_size=9, stride=4)),
                ("gdn0", GDN1(N)),
                ("conv1", conv(N, N)),
                ("gdn2", GDN1(N)),
                ("conv2", conv(N, M)),
                ("gdn3", GDN1(M)),
            ]))

        self.synthesis_transform = nn.Sequential(
            OrderedDict([
                ("igdn0", GDN1(M, inverse=True)),
                ("upsample0", nn.Upsample(scale_factor=2, mode="nearest")),
                ("convs0", conv(M, N, stride=1)),
                ("igdn1", GDN1(N, inverse=True)),
                ("upsample1", nn.Upsample(scale_factor=2, mode="nearest")),
                ("convs1", conv(N, N, stride=1)),
                ("igdn2", GDN1(N, inverse=True)),
                ("upsample2", nn.Upsample(scale_factor=4, mode="nearest")),
                ("convs2", conv(N, 3, kernel_size=9, stride=1)),
            ]))
