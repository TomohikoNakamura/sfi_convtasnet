'''Implementations of latent analog filters

Copyright (c) Tomohiko Nakamura
All rights reserved.
'''

import numpy
import torch
import torch.nn as nn


def erb_to_hz(x):
    '''Convert ERB to Hz

    Args:
        x (numpy.ndarray or float): Frequency in ERB scale

    Return:
        numpy.ndarray or float: Frequency in Hz
    '''
    return (numpy.exp(x/9.265)-1)*24.7*9.265

def hz_to_erb(x):
    '''Convert Hz to ERB

    Args:
        x (numpy.ndarray or float): Frequency in Hz

    Return:
        numpy.ndarray or float: Frequency in ERB scale
    '''
    return numpy.log(1+x/(24.7*9.265))*9.265


#############################################
class ModulatedGaussianFilters(nn.Module):
    '''Modulated Gaussian filters

        The frequency response of this filter is given by

        [
            H(\omega) = e^{-(\omega-\omega_{c})^2/(2\sigma^2)} + e^{-(\omega+\omega_{c})^2/(2\sigma^2)}.
        ]

        If one_sided is True, this frequency response is changed as

        [
            H(\omega) = e^{-(\omega-\omega_{c})^2/(2\sigma^2)}.
        ]

    '''
    def __init__(self, n_filters, init_type="erb", min_bw=1.0*2.0*numpy.pi, initial_freq_range=[50.0, 32000/2], one_sided=False, init_sigma=100.0*2.0*numpy.pi, trainable=True):
        '''

        Args:
            n_filters (int): Number of filters
            init_type (str): Initialization type of center frequencies.
                If "erb", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the ERB scale.
                If "linear", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the linear frequency scale.
            min_bw (float): Minimum bandwidth in radian
            initial_freq_range ([float,float]): Initial frequency ranges in Hz, as tuple of minimum (typically 50) and maximum values (typically, half of Nyquist frequency)
            one_sided (bool): If True, ignore the term in the negative frequency region. If False, the corresponding impulse response is modulated Gaussian window.
            init_sigma (float): Initial value for sigma
            trainable (bool): Whether filter parameters are trainable or not.
        '''
        super().__init__()
        lf, hf = initial_freq_range
        if init_type == "linear":
            mus = numpy.linspace(lf, hf, n_filters)*2.0*numpy.pi
            sigma2s = init_sigma**2 * numpy.ones((n_filters,), dtype='f')
        elif init_type == "erb":
            erb_mus = numpy.linspace(hz_to_erb(lf), hz_to_erb(hf), n_filters)
            mus = erb_to_hz(erb_mus)*2.0*numpy.pi
            sigma2s = init_sigma**2 * numpy.ones((n_filters,), dtype='f')
        else:
            raise ValueError
        self.min_ln_sigma2s = numpy.log(min_bw**2)

        self.mus = nn.Parameter(torch.from_numpy(mus).float(), requires_grad=trainable)
        self._ln_sigma2s = nn.Parameter(torch.from_numpy(numpy.log(sigma2s)).float().clamp(min=self.min_ln_sigma2s), requires_grad=trainable)
        self.phase = nn.Parameter(torch.zeros((n_filters,), dtype=torch.float), requires_grad=trainable)
        self.phase.data.uniform_(0.0, numpy.pi)
        self.one_sided = one_sided

    @property
    def sigma2s(self):
        return self._ln_sigma2s.clamp(min=self.min_ln_sigma2s).exp()

    def get_frequency_responses(self, omega: torch.Tensor):
        '''Sample frequency responses at omega

        Args:
            omega (torch.Tensor): Angular frequencies (n_angs)
        
        Return
            tuple[torch.Tensor]: Real and imaginary parts of frequency responses sampled at omega.
        '''
        if self.one_sided:
            resp_abs = torch.exp(-(omega[None,:] - self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # n_filters x n_angfreqs
            resp_r = resp_abs * self.phase.cos()[:,None]
            resp_i = resp_abs * self.phase.sin()[:,None]
        else:
            resp_abs = torch.exp(-(omega[None,:] - self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # n_filters x n_angfreqs
            resp_abs2 = torch.exp(-(omega[None,:] + self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # to ensure filters whose impulse responses are real.
            resp_r = resp_abs * self.phase.cos()[:,None] + resp_abs2 * ((-self.phase).cos()[:,None])
            resp_i = resp_abs * self.phase.sin()[:,None] + resp_abs2 * ((-self.phase).sin()[:,None])
        return resp_r, resp_i

    def extra_repr(self):
        s = f'n_filters={int(self.mus.shape[0])}, one_sided={self.one_sided}'
        return s.format(**self.__dict__)

    @property
    def device(self):
        return self.mus.device

class TDModulatedGaussianFilters(ModulatedGaussianFilters):
    def __init__(self, n_filters, train_sample_rate, init_type="erb", min_bw=1.0*2.0*numpy.pi, initial_freq_range=[50.0, 32000/2], one_sided=False, init_sigma=100.0*2.0*numpy.pi, trainable=True):
        '''

        Args:
            n_filters (int): Number of filters
            train_sample_rate (float): Trained sampling frequency
            init_type (str): Initialization type of center frequencies.
                If "erb", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the ERB scale.
                If "linear", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the linear frequency scale.
            min_bw (float): Minimum bandwidth in radian
            initial_freq_range ([float,float]): Initial frequency ranges in Hz, as tuple of minimum (typically 50) and maximum values (typically, half of Nyquist frequency)
            one_sided (bool): If True, ignore the term in the negative frequency region. If False, the corresponding impulse response is modulated Gaussian window.
            init_sigma (float): Initial value for sigma
            trainable (bool): Whether filter parameters are trainable or not.
        '''
        super().__init__(n_filters=n_filters, init_type=init_type, min_bw=min_bw, initial_freq_range=initial_freq_range, one_sided=one_sided, init_sigma=init_sigma, trainable=trainable)
        self.register_buffer('train_sample_rate', torch.tensor(float(train_sample_rate)))

    def get_impulse_responses(self, sample_rate: int, tap_size: int):
        '''Sample impulse responses

        Args:
            sample_rate (int): Target sampling frequency
            tap_size (int): Tap size
        
        Return
            torch.Tensor: Sampled impulse responses (n_filters x tap_size)
        '''
        center_freqs_in_hz = self.mus/(2.0*numpy.pi)
        # check whether the center frequencies are below Nyquist rate
        if self.train_sample_rate > sample_rate:
            mask = center_freqs_in_hz <= sample_rate/2
        ###
        t = (torch.arange(0.0, tap_size, 1).type_as(center_freqs_in_hz)/sample_rate)
        t = (t - t.mean())[None,:]
        ###
        if self.one_sided:
            raise NotImplementedError
        else:
            c = 2.0*(2.0*numpy.pi*self.sigma2s[:,None]).sqrt()*(-self.sigma2s[:,None]*(t**2)/2.0).exp()
            filter_coeffs = c*(self.mus[:,None] @ t + self.phase[:,None]).cos() # n_filters x tap_size
        if self.train_sample_rate > sample_rate:
            filter_coeffs = filter_coeffs * mask[:,None]
        return filter_coeffs[:,torch.arange(tap_size-1,-1,-1)]

#############################################
class MultiPhaseGammaToneFilters(nn.Module):
    '''Multiphase gamma tone filters

    Remark:
        This class includes the creation of Hilbert transform pairs.

    [2] D. Ditter and T. Gerkmann, ``A multi-phase gammatone filterbank for speech separation via TasNet,'' in Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing, 2020, pp. 36--40.    
    '''
    def __init__(self, n_filters, train_sample_rate, initial_freq_range=[100.0, 16000/2], n_center_freqs = 24, trainable=False):
        '''

        Args:
            n_filters (int): Number of filters
            train_sample_rate (float): Trained sampling frequency
            initial_freq_range ([float,float]): Initial frequency ranges in Hz, as tuple of minimum (typically 50) and maximum values (typically, half of Nyquist frequency)
            n_center_freqs (int): Number of center frequencies
            trainable (bool): Whether filter parameters are trainable or not.
        '''
        super().__init__()
        self.register_buffer('train_sample_rate', torch.tensor(float(train_sample_rate)))
        self.n_filters = n_filters
        assert n_filters//2 >= n_center_freqs
        ## Ditter's initialization method
        if trainable:
            self.center_freqs_in_hz = nn.Parameter(
                torch.from_numpy(erb_to_hz(numpy.linspace(hz_to_erb(initial_freq_range[0]), hz_to_erb(initial_freq_range[1]), n_center_freqs)).astype('f')).float(), # [Hz]
                requires_grad=trainable
            )
        else:
            self.register_buffer('center_freqs_in_hz', torch.from_numpy(erb_to_hz(numpy.linspace(hz_to_erb(initial_freq_range[0]), hz_to_erb(initial_freq_range[1]), n_center_freqs)).astype('f')).float())
        ###
        n_phase_variations_list = (numpy.ones(n_center_freqs)*numpy.floor(self.n_filters/2/n_center_freqs)).astype('i')
        remaining_phases = int(self.n_filters//2 - n_phase_variations_list.sum())
        if remaining_phases > 0:
            n_phase_variations_list[:remaining_phases] += 1
        n_phase_variations_list = [int(_) for _ in n_phase_variations_list]
        self.register_buffer('n_phase_variations', torch.tensor(n_phase_variations_list))
        ###
        phases = []
        for N in n_phase_variations_list:
            phases.append(numpy.linspace(0.0, numpy.pi, N))
        phases = numpy.concatenate(phases, axis=0)
        ##
        if trainable:
            self.phases = nn.Parameter(torch.from_numpy(phases).float(), requires_grad=trainable) # n_filters//2
        else:
            self.register_buffer('phases', torch.from_numpy(phases).float())
    
    def compute_gammatone_impulse_response(self, center_freqs_in_hz, phases, t):
        '''Comptue gammatone impulse responses

        Args:
            center_freqs_in_hz (torch.Tensor): Center frequencies in Hz
            phases (torch.Tensor): Phases
            sample_rate (float): Sampling frequency
        
        Return:
            torch.Tensor: Sampled impulse response (n_center_freqs x tap_size)
        '''
        center_freqs_in_hz = center_freqs_in_hz[:,None]
        n = 2
        b = (24.7 + center_freqs_in_hz/9.265) / ((numpy.pi * numpy.math.factorial(2*n-2) * numpy.power(2, float(-(2*n-2))) )/ numpy.square(numpy.math.factorial(n-1))) # equiavalent rectangular bandwidth
        a = 1.0
        return a * (t**(n-1)) * torch.exp(-2*numpy.pi*b*t) * torch.cos(2*numpy.pi*center_freqs_in_hz*t+phases[:,None]) # n_center_freqs x tap_size

    def normalize_filters(self, filter_coeffs):
        '''Normalize filter coefficients

        Args:
            filter_coeffs (torch.Tensor): Filter coefficients (n_filters x tap_size)
        
        Return:
            torch.Tensor: Normalized filter coefficients (n_filters x tap_size)
        '''
        rms_per_filter = (filter_coeffs**2).mean(dim=1).sqrt()
        C = 1.0/(rms_per_filter/rms_per_filter.max())
        return filter_coeffs * C[:,None]

    def get_impulse_responses(self, sample_rate: int, tap_size: int):
        '''Sample impulse responses

        Args:
            sample_rate (int): Target sampling frequency
            tap_size (int): Tap size
        
        Return
            torch.Tensor: Sampled impulse responses (n_filters x tap_size)
        '''
        phases = torch.cat((self.phases, self.phases+numpy.pi), dim=0) # n_filters
        center_freqs_in_hz = self.center_freqs_in_hz.repeat_interleave(self.n_phase_variations, dim=0)
        center_freqs_in_hz = center_freqs_in_hz.repeat(2) # doubles for Hilbert pairs
        # check whether the center frequencies are below Nyquist rate
        if self.train_sample_rate > sample_rate:
            mask = center_freqs_in_hz <= sample_rate/2
        ###
        if tap_size%2 == 0:
            # even: exclude the origin
            t = (torch.arange(1.0, tap_size+1, 1).type_as(center_freqs_in_hz)/sample_rate)[None,:]
        else:
            # odd: include the origin
            t = (torch.arange(0.0, tap_size, 1).type_as(center_freqs_in_hz)/sample_rate)[None,:]
        filter_coeffs = self.compute_gammatone_impulse_response(center_freqs_in_hz, phases, t).type_as(center_freqs_in_hz) # n_center_freqs x tap_size
        filter_coeffs = self.normalize_filters(filter_coeffs).type_as(center_freqs_in_hz)
        if self.train_sample_rate > sample_rate:
            filter_coeffs = filter_coeffs * mask[:,None]
        return filter_coeffs[:,torch.arange(tap_size-1,-1,-1)]
