'''Copyied from https://github.com/pfnet-research/meta-tasnet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1dWrapper(generated, *args, **kwargs):
    """
    Wrapper around the convolutional layer generated by an instrument embedding, and standard static convolution

    Arguments:
        generated {bool} -- True if you want to use the generated convolution
        *args -- Positional arguments passed to the __init__ function of the chosen module
        **kwargs -- Keyword arguments passed to the __init__ function of the chosen module

    Returns:
        nn.Module
    """
    if generated: return Conv1dGenerated(*args, **kwargs)
    else:         return Conv1dStatic(*args, **kwargs)


class Conv1dGenerated(nn.Module):
    """
    1D convolution with a kernel generated by a linear transformation of the instrument embedding
    """
    def __init__(self, E_1, E_2, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        """
        Arguments:
            E_1 {int} -- Dimension of the instrument embedding
            E_2 {int} -- Dimension of the instrument embedding bottleneck
            in_channels {int} -- Number of channels of the input
            out_channels {int} -- Number of channels of the output
            kernel_size {int} -- Kernel size of the convolution

        Keyword Arguments:
            stride {int} -- Stride of the convolution (default: {1})
            padding {int} -- Padding of the convolution (default: {0})
            dilation {int} -- Dilation of the convolution (default: {1})
            groups {int} -- Number of groups of the convolution (default: {1})
            bias {bool} -- Whether to use bias in the convolution (default: {False})
        """
        super(Conv1dGenerated, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.bottleneck = nn.Linear(E_1, E_2) if E_1 is not None else nn.Parameter(torch.randn((4, E_2)), requires_grad=True)
        self.kernel = nn.Linear(E_2, out_channels*in_channels//groups*kernel_size)
        self.bias = nn.Linear(E_2, self.out_channels) if bias else None

    def forward(self, instrument, x):
        """
        Arguments:
            instrument {torch.tensor} -- Instrument embedding of shape (4, E_1)
            x {torch.tensor} -- Input of the convolution of shape (B, 4, C, T)

        Returns:
            torch.tensor -- Output of the convolution of shape (B, 4, C', T)
        """
        batch_size = x.shape[0]

        instrument = self.bottleneck(instrument)  # shape: (4, E_2)
        kernel = self.kernel(instrument).view(4*self.out_channels, self.in_channels//self.groups, self.kernel_size)  # shape: (4*C', C//groups, kernel_size)

        # use grouped conv to process all 4 instruments independently
        x = x.view(batch_size, 4*self.in_channels, -1)  # shape: (B, 4*C, T)
        x = F.conv1d(x, kernel, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=4*self.groups)  # shape: (B, 4*C', T)
        x = x.view(batch_size, 4, self.out_channels, -1)  # shape: (B, 4, C', T)

        if self.bias:
            x += self.bias(instrument).view(1, 4, self.out_channels, 1)  # shape: (B, 4, C', T)

        return x  # shape: (B, 4, C', T)


class Conv1dStatic(nn.Module):
    """
    1D convolution with an independent kernel for each instrument
    """
    def __init__(self, _, __, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, n_src=4):
        """
        Arguments:
            in_channels {int} -- Number of channels of the input
            out_channels {int} -- Number of channels of the output
            kernel_size {int} -- Kernel size of the convolution

        Keyword Arguments:
            stride {int} -- Stride of the convolution (default: {1})
            padding {int} -- Padding of the convolution (default: {0})
            dilation {int} -- Dilation of the convolution (default: {1})
            groups {int} -- Number of groups of the convolution (default: {1})
            bias {bool} -- Whether to use bias in the convolution (default: {False})
        """
        super(Conv1dStatic, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_src = n_src
        self.conv = nn.Conv1d(n_src*in_channels, n_src*out_channels, kernel_size, stride, padding, dilation, 4*groups, bias)

    def forward(self, _, x):
        """
        Arguments:
            _ {None} -- unused argument (for compatibility with Conv1dGenerated)
            x {torch.tensor} -- Input of the convolution of shape (B, 4, C, T)

        Returns:
            torch.tensor -- Output of the convolution of shape (B, 4, C', T)
        """
        batch_size = x.shape[0]

        # use grouped conv to process all 4 instruments independently
        x = x.view(batch_size, self.n_src*self.in_channels, -1)  # shape: (B, 4*C, T)
        x = self.conv(x)  # shape: (B, 4*C', T)
        x = x.view(batch_size, self.n_src, self.out_channels, -1)  # shape: (B, 4, C', T)

        return x  # shape: (B, 4, C', T)

class Conv1dStatic2(nn.Module):
    """
    1D convolution with an independent kernel for each instrument
    """
    def __init__(self, _, __, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, n_src=4):
        """
        Arguments:
            in_channels {int} -- Number of channels of the input
            out_channels {int} -- Number of channels of the output
            kernel_size {int} -- Kernel size of the convolution

        Keyword Arguments:
            stride {int} -- Stride of the convolution (default: {1})
            padding {int} -- Padding of the convolution (default: {0})
            dilation {int} -- Dilation of the convolution (default: {1})
            groups {int} -- Number of groups of the convolution (default: {1})
            bias {bool} -- Whether to use bias in the convolution (default: {False})
        """
        super(Conv1dStatic2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_src = n_src
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        """
        Arguments:
            _ {None} -- unused argument (for compatibility with Conv1dGenerated)
            x {torch.tensor} -- Input of the convolution of shape (B, 4, C, T)

        Returns:
            torch.tensor -- Output of the convolution of shape (B, 4, C', T)
        """
        batch_size = x.shape[0]

        # use grouped conv to process all 4 instruments independently
        x = x.view(batch_size*self.n_src, self.in_channels, -1)  # shape: (B*N, C, T)
        x = self.conv(x)  # shape: (B*N, C', T)
        x = x.view(batch_size, self.n_src, self.out_channels, -1)  # shape: (B, N, C', T)

        return x  # shape: (B, 4, C', T)