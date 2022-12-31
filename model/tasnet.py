'''Implementations of SFI-ConvTasNet.

This code is based on https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh.

Copyright (c) Tomohiko Nakamura
All rights reserved.
'''
import torch
import torch.nn as nn
import warnings

from sfi_conv.fir_design import FreqRespSampConv1d, ImpRespSampConv1d, FreqRespSampConvTranspose1d, ImpRespSampConvTranspose1d
from model.mask_tcn import MaskingModule
from utility.loss import calculate_loss

###################
class ApproxFIRConv1d(nn.Module):
    """Encodes a waveform into a latent representation.

    [1] K. Saito, T. Nakamura, K. Yatabe, and H. Saruwatari, ``Sampling-frequency-independent convolutional layer and its application to audio source separation,'' IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 2928--2943, Sep. 2022.
    """
    def __init__(self, fir_computation_type, ContFilterType, filter_params, N, nonlineartity="relu", n_samples_freq_resp=640, use_Hilbert_transforms=False):
        """
        
        Args:
            fir_computation_type (str): Specify the FIR design techniques (frequency domain [freq_resp_samp] or time domain [imp_resp_samp]
            ContFilterType (Class): Class name of latent analog filter
            filter_params (dict): Parameters for latent analog filter
            N (int): The number of channels
            nonlinearity (str): Nonlinearity for output of convolutional layer (None or "relu")
            n_samples_freq_resp (int): The number of sampled points of frequency responses
            use_Hilbert_transforms (bool): If True, Hilbert transforms are used.
        """
        super(ApproxFIRConv1d, self).__init__()
        self._require_prepare = True
        self._fir_computation_type = fir_computation_type
        if fir_computation_type == "freq_resp_samp":
            self.conv = FreqRespSampConv1d(1, N, n_samples=n_samples_freq_resp, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms)
            self._require_prepare = True
        elif fir_computation_type == "imp_resp_samp":
            self.conv = ImpRespSampConv1d(1, N, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms)
            self._require_prepare = True
        if nonlineartity is None:
            self.nonlinearity = nn.Sequential()
        elif nonlineartity == "relu":
            self.nonlinearity = nn.ReLU(True)
        else:
            raise ValueError(f'Unknown nonlinearity [{nonlineartity}]')

    def convert(self):
        '''Convert an SFI (tranposed) convolutional layer into the forward function of an usual (transposed) convolutional layer
        
        Remark: Once this function is called, the converted usual convolutional layer (i.e., an usual convolutional layer for a specific sampling frequency) is always used. If you adapt it to another sampling frequency, call the `convert` function again.
        '''
        if self._require_prepare:
            self.converted_conv = self.conv.convert()

    @property
    def is_SFI(self):
        '''Whether this layer is SFI or not.
        '''
        return self._require_prepare

    def prepare(self, sample_rate:int, kernel_size: int, stride: int, padding: int=None):
        '''Prepare for sampling latent analog filters

        Args:
            sample_rate (int): Sampling frequency
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding
        '''
        if self._require_prepare:
            self.conv.prepare(sample_rate, kernel_size, stride, padding=padding)
        
    def change_n_samples(self, n_samples):
        '''Change the number of samples for sampling frequency responses

        Args:
            n_samples (int): The number of sampled points for frequency sampling
        '''
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        self.conv.n_samples = n_samples

    def get_n_samples(self):
        '''Return the number of samples for sampling frequency responses

        Return:
            int: The number of sampled points for frequency sampling
        '''
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        return self.conv.n_samples

    def forward(self, signal):
        """

        Args:
            signal (torch.Tensor): Mixture signal (batch x 1 x time)

        Returns:
            torch.Tensor: Pseudo latent representation (batch x n_channels x time)
        """
        return self.nonlinearity(self.converted_conv(signal) if hasattr(self, "converted_conv") else self.conv(signal))

class ApproxFIRConvTranspose1d(nn.Module):
    """Decodes the latent representation back to waveforms

    [1] K. Saito, T. Nakamura, K. Yatabe, and H. Saruwatari, ``Sampling-frequency-independent convolutional layer and its application to audio source separation,'' IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 2928--2943, Sep. 2022.
    """
    def __init__(self, fir_computation_type, ContFilterType, filter_params, N, n_samples_freq_resp=None, use_Hilbert_transforms=False):
        """
        Arguments:
            fir_computation_type (str): Specify the FIR design techniques (frequency domain [freq_resp_samp] or time domain [imp_resp_samp]
            ContFilterType (Class): Class name of latent analog filter
            filter_params (dict): Parameters for latent analog filter
            N (int): The number of channels
            n_samples_freq_resp (int): The number of sampled points of frequency responses
            use_Hilbert_transforms (bool): If True, Hilbert transforms are used.
        """
        super(ApproxFIRConvTranspose1d, self).__init__()
        self._require_prepare = True
        self._fir_computation_type = fir_computation_type
        if fir_computation_type == "freq_resp_samp":
            self.conv = FreqRespSampConvTranspose1d(N, 1, n_samples=n_samples_freq_resp, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms)
            self._require_prepare = True
        elif fir_computation_type == "imp_resp_samp":
            self.conv = ImpRespSampConvTranspose1d(N, 1, ContFilterType=ContFilterType, filter_params=filter_params, use_Hilbert_transforms=use_Hilbert_transforms)
            self._require_prepare = True
        else:
            raise NotImplementedError

    def convert(self):
        '''Convert an SFI (tranposed) convolutional layer into the forward function of an usual (transposed) convolutional layer
        
        Remark: Once this function is called, the converted usual convolutional layer (i.e., an usual convolutional layer for a specific sampling frequency) is always used. If you adapt it to another sampling frequency, call the `convert` function again.
        '''
        if self._require_prepare:
            self.converted_conv = self.conv.convert()

    @property
    def is_SFI(self):
        '''Whether this layer is SFI or not.
        '''
        return self._require_prepare

    def prepare(self, sample_rate:int, kernel_size: int, stride: int, padding: int=None, output_padding: int=0):
        '''Prepare for sampling latent analog filters

        Args:
            sample_rate (int): Sampling frequency
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding
        '''
        if self._require_prepare:
            self.conv.prepare(sample_rate, kernel_size, stride, padding=padding, output_padding=output_padding)

    def change_n_samples(self, n_samples):
        '''Change the number of samples for sampling frequency responses

        Args:
            n_samples (int): The number of sampled points for frequency sampling
        '''
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        self.conv.n_samples = n_samples

    def get_n_samples(self):
        '''Return the number of samples for sampling frequency responses

        Return:
            int: The number of sampled points for frequency sampling
        '''
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        return self.conv.n_samples

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): Pseudo latent representations (batch x n_channels x time)

        Returns:
            torch.Tensor: Signal (batch x 1 x time)
        """
        return self.converted_conv(x) if hasattr(self, "converted_conv") else self.conv(x)

###################
class VariableFIRConvTasNet(nn.Module):
    """Variable sampling frequency Conv-TasNet
    One stage of encoder->mask->decoder for a single sampling rate
    """
    def __init__(self, encoder_params, decoder_params, N, L, W, B, H, sample_rate, partial_input, args):
        """
        Arguments:
            independent_params {bool} -- False if you want to use the generated weights
            N {int} -- # of encoder filters
            L {int} -- Kernel size of encoder
            W {int} -- Encoder stride
            B {int} -- Dimension of the bottleneck convolution in the masking subnetwork
            H {int} -- Hidden dimension of the masking subnetwork
            sr {int} -- Sampling rate of the processed signal
            partial_input {bool} -- True if the module should expect input from preceding stage
            args {dict} -- Other argparse hyperparameters
        """
        super(VariableFIRConvTasNet, self).__init__()

        self.N = N
        self.stride = W
        self.out_channels = 1
        self.C = 4

        self._frame = L
        self._stride = W
        self.encoder = ApproxFIRConv1d(N=N, **encoder_params)
        self.decoder = ApproxFIRConvTranspose1d(N=N, **decoder_params)

        self.dropout = nn.Dropout2d(args.dropout)
        self.mask = MaskingModule(False, 0, 0, N, B, H, args.layers, args.stack, args.kernel, args.residual_bias, partial_input=partial_input)

        # Initialize analog filters
        self.prepare(sample_rate, self._frame, self._stride, force=True)

        self.args = args

    @property
    def is_SFI(self):
        return self.encoder.is_SFI and self.decoder.is_SFI

    def convert(self):
        '''Convert SFI convolutional layer into usual convolutional layer at specified sampling frequency for faster inference.

        WARNING: This function must not be called during training and validation. After this function is called, the SFI convolutional layer always uses the precomputed weights, of which sampling frequency is specified when the function is called. Since the computation of the weight generation is omitted, this makes the inference faster but disable the model to adapt to another sampling frequencies.
        '''
        self.encoder.convert()
        self.decoder.convert()

    def prepare(self, sample_rate:int, kernel_size:int, stride:int, force: bool=False):
        if self.is_SFI or force:
            # Both encoder and decoder are SFI.
            self.encoder.prepare(sample_rate, kernel_size, stride)
            self.decoder.prepare(sample_rate, kernel_size, stride)
        else:
            # Do nothing when at least either of encoder and decoder is not SFI.
            warnings.warn('Do not adapt the model to the target sampling frequency due to the lack of the SFI property.')

    def forward(self, input_mix, separated_inputs, mask, partial_input=None):
        """
        Forward pass for training; returns the loss and hidden state to be passed to the next stage

        Arguments:
            input_mix {torch.tensor} -- Mixed signal of shape (B, 1, T)
            separated_inputs {torch.tensor} -- Ground truth separated mixed signal of shape (B, 4, 1, T)
            mask {torch.tensor} -- Boolean mask: True when $separated_inputs is 0.0; shape: (B, 4, 1)

        Keyword Arguments:
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T') (default: {None})

        Returns:
            (torch.tensor, torch.tensor, torch.tensor) -- (
                the total loss of shape (),
                list of statistics with partial losses and metrics of shape (7),
                partial input to be passed to the next stage of shape (B, 4, N, T')
            )
        """
        batch_size = input_mix.shape[0]

        # waveform encoder
        mix_latent = self.encoder(input_mix)  # shape: (B, N, T')
        mix_latents = mix_latent.unsqueeze(1)  # shape: (B, 1, N, T')
        mix_latents = mix_latents.expand(-1, self.C, -1, -1).contiguous()  # shape: (B, 4, N, T')

        if self.args.similarity_loss_weight > 0.0 or self.args.dissimilarity_loss_weight > 0.0:
            separated_gold_latents = self.encoder(separated_inputs.view(self.C*batch_size, input_mix.shape[1], -1))  # shape: (B*4, N, T')
            separated_gold_latents = separated_gold_latents.view(batch_size, self.C, self.N, -1).permute(0, 1, 3, 2).contiguous()  # shape: (B, 1, T', N)
        else:
            separated_gold_latents = None

        # generate masks
        mask_input = self.dropout(mix_latents.view(batch_size*self.C, self.N, -1).unsqueeze(-1)).squeeze(-1).view(batch_size, self.C, self.N, -1)  # shape: (B, 4, N, T')
        masks = self.mask(None, mask_input, partial_input)  # shape: (B, 4, N, T')

        separated_latents = mix_latents * masks  # shape: (B, 4, N, T')

        # waveform decoder
        decoder_input = separated_latents.view(batch_size * self.C, self.N, -1)  # shape: (B*4, N, T')
        output_signal = self.decoder(decoder_input)  # shape: (B*4, channels, T)
        output_signal = output_signal.view(batch_size, self.C, self.out_channels, -1)  # shape: (B, 4, 1, T) [drums, bass, other, vocals]

        if self.args.reconstruction_loss_weight > 0:
            reconstruction = self.decoder(mix_latent)  # shape: (B, 1, T)
        else:
            reconstruction = None
        loss, stats = calculate_loss(output_signal, separated_inputs, mask, separated_gold_latents, reconstruction, input_mix, self.args)
        return loss, stats, separated_latents

    def inference(self, x, partial_input=None):
        """
        Forward pass for inference; returns the separated signal and hidden state to be passed to the next stage

        Arguments:
            x {torch.tensor} -- mixed signal of shape (1, 1, T)

        Keyword Arguments:
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T') (default: {None})

        Returns:
            (torch.tensor, torch.tensor) -- (
                separated signal of shape (1, 4, 1, T),
                hidden state to be passed to the next stage of shape (1, 4, N, T')
            )
        """

        x = self.encoder(x)  # shape: (1, N, T')
        x = x.expand(self.C, -1, -1).unsqueeze_(0)  # shape: (1, 4, N, T')

        if partial_input is not None:
            mask_input = torch.cat([x, partial_input], 2)  # shape: (1, 4, N+N/2, T')
        else:
            mask_input = x  # shape: (1, 4, N, T')
        del partial_input

        masks = self.mask(None, mask_input)  # shape: (1, 4, N, T')
        del mask_input

        x = x * masks  # shape: (1, 4, N, T')
        del masks

        x.squeeze_(0)  # shape: (4, N, T')
        hidden = x

        x = self.decoder(x)  # shape: (4, 1, T)

        return x.unsqueeze_(0), hidden.unsqueeze_(0)  # shape: [(1, 4, 1, T), (1, 4, N, T')]

class MultiVariableFIRConvTasNet(nn.Module):
    """
    Multiple stages of Tasnet stacked sequentially
    """
    def __init__(self, args):
        """
        Arguments:
            args {dict} -- Other argparse hyperparameters
        """
        super(MultiVariableFIRConvTasNet, self).__init__()

        self.args = args
        self.W = args.W
        self.L = args.L
        self.base_sr = args.sampling_rate
        self.stages_num = args.stages_num

        self.stage = VariableFIRConvTasNet(
            encoder_params=args.encoder_params,
            decoder_params=args.decoder_params,
            N = args.N, L = args.L, W = args.W, B = args.B, H = args.H, 
            sample_rate = args.sampling_rate, partial_input=False, 
            args=args
        )
    
    def convert(self):
        self.stage.convert()

    @property
    def is_SFI(self):
        return self.stage.is_SFI

    def forward(self, input_mixes, separated_inputs, masks):
        """
        Forward pass for training

        Arguments:
            input_mixes {[torch.tensor]} -- List of mixed signals for all stages of shape (B, 1, T)
            separated_inputs {[torch.tensor]} -- List of ground truth separated mixed signal of shape (B, 4, 1, T)
            masks {[torch.tensor]} -- List of boolean mask: True when $separated_inputs is 0.0; shape: (B, 4, 1)

        Returns:
            (torch.tensor, torch.tensor) -- (
                the total loss of shape (1),
                list of statistics with partial losses and metrics (15)
            )
        """
        assert len(input_mixes) == self.stages_num
        assert len(separated_inputs) == self.stages_num
        assert len(masks) == self.stages_num

        loss, stats, hidden = None, None, None
        for i in range(self.stages_num):
            m = 2**i
            sample_rate = int(self.base_sr * m)
            kernel_size = int(self.L*m)
            stride = int(self.W*m)
            self.stage.prepare(sample_rate, kernel_size, stride)
            _loss, _stats, hidden = self.stage(input_mixes[i], separated_inputs[i], masks[i])

            loss = _loss if loss is None else loss + _loss
            stats = _stats if stats is None else torch.cat([stats[:i*4], _stats], 0)

        stats.unsqueeze_(0)
        loss.unsqueeze_(0)

        return loss, stats

    def inference(self, input_audio, n_chunks=4):
        """
        Forward pass for inference; returns the separated signal

        Arguments:
            input_audio {torch.tensor} -- List of mixed signals for all stages of shape (B, 1, T)

        Keyword Arguments:
            n_chunks {int} -- Divide the $input_audio to chunks to trade speed for memory (default: {4})

        Returns:
            torch.tensor -- Separated signal of shape (1, 4, 1, T)
        """
        assert len(input_audio) == self.stages_num

        time_length = self.args.time_length # 8

        # split the input audio to $n_chunks and make sure they overlap to not lose the accuracy
        # $chunk_intervals contain the (start, end) times of all chunks in a list
        chunks = [int(input_audio[0].shape[-1] / n_chunks * c + 0.5) for c in range(n_chunks)]
        chunks.append(input_audio[0].shape[-1])
        chunk_intervals = [(max(0, chunks[n] - int(self.base_sr*time_length)), min(chunks[n+1] + int(self.base_sr*time_length), input_audio[0].shape[-1])) for n in range(n_chunks)]
        chunk_intervals = [(s, e - ((e-s) % self.W)) if s == 0 else (s + (e-s) % self.W, e) for s, e in chunk_intervals]

        full_outputs = None
        for c in range(n_chunks):
            outputs = []
            for i in range(self.stages_num):
                m = 2**i
                sample_rate = int(self.base_sr * m)
                kernel_size = int(self.L*m)
                stride = int(self.W*m)
                self.stage.prepare(sample_rate, kernel_size, stride)
                output, _ = self.stage.inference(input_audio[i][:, :, m*chunk_intervals[c][0]: m*chunk_intervals[c][1]])

                output = output[:, :, :, m*(chunks[c] - chunk_intervals[c][0]): output.shape[-1] - m*((chunk_intervals[c][1] - chunks[c+1]))]
                outputs.append(output)

            # concatenate the chunks togerther
            if full_outputs is None:
                full_outputs = outputs
            else:
                full_outputs = [torch.cat([f, o], -1) for f, o in zip(full_outputs, outputs)]

        return full_outputs

    def _compute_equivalent_frame_size_and_shift(self, sample_rate):
        if not self.is_SFI:
            return self.L, self.W
        else:
            factor = sample_rate/float(self.base_sr)
            L = int(self.L*factor)
            W = int(self.W*factor)
            return L, W

    def sr_specified_inference(self, input_audio, sample_rate, n_chunks=4):
        """
        Forward pass for inference; returns the separated signal

        Arguments:
            input_audio {torch.tensor} -- List of mixed signals for all stages of shape (B, 1, T)

        Keyword Arguments:
            n_chunks {int} -- Divide the $input_audio to chunks to trade speed for memory (default: {4})

        Returns:
            torch.tensor -- Separated signal of shape (1, 4, 1, T)
        """
        assert len(input_audio) == self.stages_num

        time_length = self.args.time_length # 8

        # set L, W, and sr
        L, W = self._compute_equivalent_frame_size_and_shift(sample_rate)

        # split the input audio to $n_chunks and make sure they overlap to not lose the accuracy
        # $chunk_intervals contain the (start, end) times of all chunks in a list
        chunks = [int(input_audio[0].shape[-1] / n_chunks * c + 0.5) for c in range(n_chunks)]
        chunks.append(input_audio[0].shape[-1])
        chunk_intervals = [(max(0, chunks[n] - int(sample_rate*time_length)), min(chunks[n+1] + int(sample_rate*time_length), input_audio[0].shape[-1])) for n in range(n_chunks)]
        chunk_intervals = [(s, e - ((e-s) % W)) if s == 0 else (s + (e-s) % W, e) for s, e in chunk_intervals]

        full_outputs = None
        for c in range(n_chunks):
            outputs = []
            for i in range(self.stages_num):
                m = 2**i
                sample_rate = int(sample_rate * m)
                kernel_size = int(L*m)
                stride = int(W*m)
                self.stage.prepare(sample_rate, kernel_size, stride)
                output, _ = self.stage.inference(input_audio[i][:, :, m*chunk_intervals[c][0]: m*chunk_intervals[c][1]])

                output = output[:, :, :, m*(chunks[c] - chunk_intervals[c][0]): output.shape[-1] - m*((chunk_intervals[c][1] - chunks[c+1]))]
                outputs.append(output)

            # concatenate the chunks togerther
            if full_outputs is None:
                full_outputs = outputs
            else:
                full_outputs = [torch.cat([f, o], -1) for f, o in zip(full_outputs, outputs)]

        return full_outputs
