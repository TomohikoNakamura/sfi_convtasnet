'''Implementations of frequency- and time-domain filter designs using latent analog filters

Copyright (c) Tomohiko Nakamura
All rights reserved.
'''
import functools
import warnings
from distutils.version import LooseVersion

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single

from sfi_conv import continuous_filters as cf_module


def get_continous_filter(ContFilterType, n_filters, filter_params):
    if hasattr(cf_module, ContFilterType):
        continuous_filters = getattr(cf_module, ContFilterType)(n_filters=n_filters, **filter_params)
    else:
        raise NotImplementedError(f'Undefined Filter Type [{ContFilterType}]')
    return continuous_filters

# common modules
def compute_Hilbert_transforms_of_filters(filters):
    '''Compute the Hilber transforms of the input filters

    Args:
        filters (torch.Tensor): Filters (n_filters x kernel_size)

    Return
        torch.Tensor: Hilbert transforms of the weights (out_channels x in_channels x kernel_size)
    '''
    if LooseVersion(torch.__version__) < LooseVersion("1.7.0"):
        ft_f = torch.rfft(filters.reshape(filters.shape[0], 1, filters.shape[1]), 1, normalized=True)
        hft_f = torch.stack([ft_f[:, :, :, 1], - ft_f[:, :, :, 0]], dim=-1)
        hft_f = torch.irfft(hft_f, 1, normalized=True, signal_sizes=(filters.shape[1],))
    else: # New PyTorch version has fft module
        ft_f = torch.fft.rfft(filters, n=filters.shape[1], dim=1, norm="ortho")
        hft_f = torch.view_as_complex(torch.stack((ft_f.imag, -ft_f.real), axis=-1))
        hft_f = torch.fft.irfft(hft_f, n=filters.shape[1], dim=1, norm="ortho")
    return hft_f.reshape(*(filters.shape))

class _FIRDesignBase(nn.Module):
    def __init__(self, in_channels, out_channels, ContFilterType, filter_params, use_Hilbert_transforms=False, transposed=False):
        '''

        Args:
            in_channels (int): Number of channels of 1D sequence
            out_channels (int): Number of channels produced by the convolution
            ContFilterType (Class): Latent analog filter class
            filter_params (dict): Parameters of latent analog filter class
            use_Hilbert_transforms (bool): If True, the latter half of the filters are the Hilbert pairs of the former half.
            transposed (bool): Whether this convolution is a transposed convolution or not.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        n_filters = in_channels*out_channels
        self.use_Hilbert_transforms = use_Hilbert_transforms
        if self.use_Hilbert_transforms:
            if n_filters%2 == 1:
                raise ValueError(f'n_filters must be even when using Hilbert transforms of filters [n_filters={n_filters}]')
            n_filters //= 2
        self.continuous_filters = get_continous_filter(ContFilterType=ContFilterType, n_filters=n_filters, filter_params=filter_params)
        self._transposed = transposed

    def weights(self, filters):
        '''Return weights

        Args:
            filters (torch.Tensor): Filters (n_channels x tap_size)

        Return:
            torch.Tensor: Weights. The shape is in_channel x out_channel x tap_size for an SFI convolutional layer and out_channel x in_channel x tap_size for an SFI transposed convolutional layer.
        '''
        if self.use_Hilbert_transforms:
            filters = torch.cat((filters, compute_Hilbert_transforms_of_filters(filters)), dim=0)
        if self._transposed:
            return filters.reshape(self.in_channels, self.out_channels, -1)
        else:
            return filters.reshape(self.out_channels, self.in_channels, -1)

    def prepare(self, sample_rate: int, kernel_size: int, stride: int, padding: int=None, output_padding: int=0):
        '''Prepare for sampling latent analog filters

        Args:
            sample_rate (int): Sampling frequency
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding
            output_padding (int): Output padding for a tranposed convolutional layer
        '''
        self.sample_rate = sample_rate
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        if padding is None:
            self.padding = _single((self.kernel_size[0]-self.stride[0])//2)
        else:
            self.padding = _single(padding)
        self.output_padding = (int(output_padding),)

    def forward(self, input):
        raise NotImplementedError

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, sample_rate={sample_rate}'
        if hasattr(self, "kernel_size"):
            s += ', kernel_size={kernel_size}'
        if hasattr(self, 'stride'):
            s += ', stride={stride}'
        if hasattr(self, "kernel_size"):
            s += ', padding={padding}'
        if hasattr(self, "output_padding"):
            s += ', output_padding={output_padding}'
        return s.format(**self.__dict__)

    def convert(self):
        '''Convert an SFI (tranposed) convolutional layer into the forward function of an usual (transposed) convolutional layer
        
        This function returns the forward function of an usual convolutional layer that is converted from the SFI convolutional layer for fast inference at a single sampling frequency.

        Return:
            function: Forward function of an usual convolutional layer that is converted from the SFI convolutional layer for fast inference at a single sampling frequency.
        '''
        warnings.warn(f'Converting SFI to Non-SFI convolutional layers [sample_rate={self.sample_rate}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, output_padding={self.output_padding}]')
        if self.is_transposed:
            return functools.partial(F.conv_transpose1d, weight=self.weights(), bias=None, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=_single(1), groups=1)
        else:
            return functools.partial(F.conv1d, weight=self.weights(), bias=None, stride=self.stride, padding=self.padding, dilation=_single(1), groups=1)

## Frequency domain method
class _FreqRespSampFIRs(_FIRDesignBase):
    '''Base class for SFI convolutional layer using the frequendy domain filter design
    '''
    def __init__(self, in_channels, out_channels, n_samples, ContFilterType, filter_params, use_Hilbert_transforms=False, transposed=False):
        '''

        Args:
            in_channels (int): Number of channels of 1D sequence
            out_channels (int): Number of channels produced by the convolution
            n_samples (int): Number of sampled points for frequency sampling
            ContFilterType (Class): Latent analog filter class
            filter_params (dict): Parameters of latent analog filter class
            use_Hilbert_transforms (bool): If True, the latter half of the filters are the Hilbert pairs of the former half.
            transposed (bool): Whether this convolution is a transposed convolution or not.
        '''
        super().__init__(in_channels=in_channels, out_channels=out_channels, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms, transposed=transposed)
        self.n_samples = n_samples
        self._cache = dict()

    def weights(self):
        '''Return weights

        Return:
            torch.Tensor: Weights. The shape is in_channel x out_channel x tap_size for an SFI convolutional layer and out_channel x in_channel x tap_size for an SFI transposed convolutional layer.
        '''
        filters = self.approximate_by_FIR(self.continuous_filters.device) # n_filters (or n_filters//2) x kernel_size
        return super().weights(filters)

    def _compute_pinvW(self, device):
        kernel_size = self.kernel_size[0]
        sample_rate = self.sample_rate            
        P = (kernel_size-1)//2 if kernel_size%2 == 1 else kernel_size//2
        M = self.n_samples
        nyquist_rate = sample_rate / 2
        #
        ang_freqs = torch.linspace(0, nyquist_rate*2.0*numpy.pi, M).float().to(device)
        normalized_ang_freqs = ang_freqs / float(sample_rate)
        if kernel_size%2 == 1:
            seq_P = torch.arange(-P, P+1).float()[None,:].to(device)
            ln_W = -normalized_ang_freqs[:,None]*seq_P # M x 2P+1
        else:
            seq_P = torch.arange(-(P-1), P+1).float()[None,:].to(device)
            ln_W = -normalized_ang_freqs[:,None]*seq_P # M x 2P
        ln_W = ln_W.to(device)
        W = torch.cat((torch.cos(ln_W), torch.sin(ln_W)), dim=0) # 2*M x 2P
        ###
        pinvW = torch.pinverse(W) # 2P x 2M
        pinvW.requires_grad_(False)
        ang_freqs.requires_grad_(False)
        return ang_freqs, pinvW

    def approximate_by_FIR(self, device):
        '''Approximate frequency responses of analog filters with those of digital filters

        Args:
            device (torch.Device): Computation device
        
        Return:
            torch.Tensor: Time-reversed impulse responses of digital filters (n_filters x filter_degree (-P to P))
        '''
        cache_tag = (self.sample_rate, self.kernel_size, self.stride)
        if cache_tag in self._cache:
            ang_freqs, pinvW = self._cache[cache_tag]
            ang_freqs = ang_freqs.detach().to(device)
            pinvW = pinvW.detach().to(device)
        else:
            ang_freqs, pinvW = self._compute_pinvW(device)
            self._cache[cache_tag] = (ang_freqs.detach().cpu(), pinvW.detach().cpu())
        ###
        resp_r, resp_i = self.continuous_filters.get_frequency_responses(ang_freqs) # n_filters x M
        resp = torch.cat((resp_r, resp_i), dim=1) # n_filters x 2M
        ###
        fir_coeffs = (pinvW[None,:,:] @ resp[:,:,None])[:,:,0] # n_filters x 2P
        return fir_coeffs[:,torch.arange(self.kernel_size[0]-1,-1,-1)] # time-reversed impulse response

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, sample_rate={sample_rate}, n_samples={n_samples}'
        if hasattr(self, "kernel_size"):
            s += ', kernel_size={kernel_size}'
        if hasattr(self, 'stride'):
            s += ', stride={stride}'
        if hasattr(self, "kernel_size"):
            s += ', padding={padding}'
        return s.format(**self.__dict__)

    def get_analog_freqresp_for_visualization(self, ang_freqs):
        '''Get frequency responses of latent analog filters for visualization

        Args:
            ang_freqs (torch.Tensor): Unnormalized angular frequency [rad] (n_angfreqs)
        
        Return:
            torch.Tensor[cfloat]: Frequency responses of analog filters (n_filters x n_angfreqs)
        '''
        resp_r, resp_i = self.continuous_filters.get_frequency_responses(ang_freqs)
        resp = torch.stack((resp_r, resp_i), axis=-1)
        return torch.view_as_complex(resp)


class FreqRespSampConv1d(_FreqRespSampFIRs):
    '''SFI convolutional layer using the frequency domain filter design
    '''
    def __init__(self, in_channels, out_channels, n_samples, ContFilterType, filter_params, use_Hilbert_transforms=False):
        '''

        Args:
            in_channels (int): Number of channels of 1D sequence
            out_channels (int): Number of channels produced by the convolution
            n_samples (int): Number of sampled points for frequency sampling
            ContFilterType (Class): Latent analog filter class
            filter_params (dict): Parameters of latent analog filter class
            use_Hilbert_transforms (bool): If True, the latter half of the filters are the Hilbert pairs of the former half.
        '''
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_samples=n_samples, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms, transposed=False)
    
    def forward(self, input):
        '''

        Args:
            input (torch.Tensor): Input feature (batch x in_channel x time)
        
        Return:
            torch.Tensor: Output feature (batch x out_channel x time)
        '''
        return F.conv1d(input, self.weights(), None, self.stride, self.padding, _single(1), 1)
    
    @property
    def is_transposed(self):
        '''Returns whether this layer is a transposed version.
        '''
        return False

class FreqRespSampConvTranspose1d(_FreqRespSampFIRs):
    '''SFI transposed convolutional layer using the frequency domain filter design
    '''
    def __init__(self, in_channels, out_channels, n_samples, ContFilterType, filter_params, use_Hilbert_transforms=False):
        '''

        Args:
            in_channels (int): Number of channels of 1D sequence
            out_channels (int): Number of channels produced by the convolution
            n_samples (int): Number of sampled points for frequency sampling
            ContFilterType (Class): Latent analog filter class
            filter_params (dict): Parameters of latent analog filter class
            use_Hilbert_transforms (bool): If True, the latter half of the filters are the Hilbert pairs of the former half.
        '''
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_samples=n_samples, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms, transposed=True)

    def forward(self, input):
        '''

        Args:
            input (torch.Tensor): Input feature (batch x in_channel x time)
        
        Return:
            torch.Tensor: Output feature (batch x out_channel x time)
        '''
        return F.conv_transpose1d(input, self.weights(), None, self.stride, self.padding, self.output_padding, 1, _single(1))

    @property
    def is_transposed(self):
        '''Returns whether this layer is a transposed version.
        '''
        return True

## Time domain method
class _ImpRespSampFIRs(_FIRDesignBase):
    def weights(self):
        '''Return weights

        Return:
            torch.Tensor: Weights. The shape is in_channel x out_channel x tap_size for an SFI convolutional layer and out_channel x in_channel x tap_size for an SFI transposed convolutional layer.
        '''
        filters = self.continuous_filters.get_impulse_responses(self.sample_rate, self.kernel_size[0])
        return super().weights(filters)

    def get_analog_impulse_resp_for_visualization(self, sample_rate, kernel_size):
        '''Get impulse responses of latent analog filters for visualization

        Args:
            sample_rate (int): Sampling frequency
            kernel_size (int): Kernel size
        
        Return:
            torch.Tensor[cfloat]: Frequency responses of analog filters (n_filters x n_angfreqs)
        '''
        return self.continuous_filters.get_impulse_responses(sample_rate, kernel_size)

class ImpRespSampConv1d(_ImpRespSampFIRs):
    def __init__(self, in_channels, out_channels, ContFilterType, filter_params, use_Hilbert_transforms=False):
        '''

        Args:
            in_channels (int): Number of channels of 1D sequence
            out_channels (int): Number of channels produced by the convolution
            n_samples (int): Number of sampled points for frequency sampling
            ContFilterType (Class): Latent analog filter class
            filter_params (dict): Parameters of latent analog filter class
            use_Hilbert_transforms (bool): If True, the latter half of the filters are the Hilbert pairs of the former half.
        '''
        super().__init__(in_channels=in_channels, out_channels=out_channels, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms, transposed=False)
    
    def forward(self, input):
        '''

        Args:
            input (torch.Tensor): Input feature (batch x in_channel x time)
        
        Return:
            torch.Tensor: Output feature (batch x out_channel x time)
        '''
        return F.conv1d(input, self.weights(), None, self.stride, self.padding, _single(1), 1)

    @property
    def is_transposed(self):
        '''Returns whether this layer is a transposed version.
        '''
        return False

class ImpRespSampConvTranspose1d(_ImpRespSampFIRs):
    def __init__(self, in_channels, out_channels, ContFilterType, filter_params, use_Hilbert_transforms=False):
        '''

        Args:
            in_channels (int): Number of channels of 1D sequence
            out_channels (int): Number of channels produced by the convolution
            n_samples (int): Number of sampled points for frequency sampling
            ContFilterType (Class): Latent analog filter class
            filter_params (dict): Parameters of latent analog filter class
            use_Hilbert_transforms (bool): If True, the latter half of the filters are the Hilbert pairs of the former half.
        '''
        super().__init__(in_channels=in_channels, out_channels=out_channels, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms, transposed=True)
    
    def forward(self, input):
        '''

        Args:
            input (torch.Tensor): Input feature (batch x in_channel x time)
        
        Return:
            torch.Tensor: Output feature (batch x out_channel x time)
        '''
        return F.conv_transpose1d(input, self.weights(), None, self.stride, self.padding, self.output_padding, 1, _single(1))

    @property
    def is_transposed(self):
        '''Returns whether this layer is a transposed version.
        '''
        return True
