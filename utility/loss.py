import torch
import torch.nn.functional as F
import functools

def compute_STFT(signal, stft_params):
    shape = signal.shape
    signal = signal.view(shape[0]*shape[1]*shape[2], shape[3])
    STFT = torch.stft(signal, **stft_params)
    return STFT.reshape(shape[0], shape[1], shape[2], *STFT.shape[1:])

def STFT_mse_objective(estimation, origin, _, n_fft=4096, hop_length=2048, center=True, normalized=False, onesided=True, pad_mode="reflect"):
    window = torch.hann_window(n_fft).cuda()
    shape = estimation.shape
    stft_params = dict(n_fft=n_fft, hop_length=hop_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided)
    STFT_estimation = compute_STFT(estimation, stft_params)
    STFT_origin = compute_STFT(origin, stft_params)
    return F.mse_loss(STFT_estimation, STFT_origin, reduction="none").mean(dim=(0,2,3,4,5)) # shape: (4)

def STFT_mae_objective(estimation, origin, _, n_fft=4096, hop_length=2048, center=True, normalized=False, onesided=True, pad_mode="reflect"):
    window = torch.hann_window(n_fft).cuda()
    shape = estimation.shape
    stft_params = dict(n_fft=n_fft, hop_length=hop_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided)
    STFT_estimation = compute_STFT(estimation, stft_params)
    STFT_origin = compute_STFT(origin, stft_params)
    return F.l1_loss(STFT_estimation, STFT_origin, reduction="none").mean(dim=(0,2,3,4,5)) # shape: (4)

def multiresolution_STFT_objective(estimation, origin, n_ffts=[1024, 2048, 512], hop_lengths=[120, 240, 50], win_lengths=[600, 1200, 240], center=True, normalized=False, onesided=True, pad_mode="reflect", eps=1e-8):
    spectral_convergence_loss = torch.zeros(estimation.shape[:3], device=estimation.device)
    log_mag_stft_loss = torch.zeros(estimation.shape[:3], device=estimation.device)
    for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
        window = torch.hann_window(win_length).to(estimation.device)
        #
        stft_params = dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided)
        STFT_estimation = compute_STFT(estimation, stft_params) # batch x source x channel x frequency x frame x 2 (real, imag)
        STFT_origin = compute_STFT(origin, stft_params)
        mag_estimation = (STFT_estimation**2).sum(dim=5).clamp(min=eps).sqrt() # batch x source x channel x frequency x frame
        mag_origin = (STFT_origin**2).sum(dim=5).clamp(min=eps).sqrt() # batch x source x channel x frequency x frame
        # spectral convergence: batch x source x channel
        spectral_convergence_loss = spectral_convergence_loss + torch.norm(mag_origin - mag_estimation, p="fro", dim=[3,4]) / torch.norm(mag_origin, p="fro", dim=[3,4])
        # log_mag_stft_loss: batch x source x channel
        log_mag_stft_loss = log_mag_stft_loss + F.l1_loss(mag_origin.log(), mag_estimation.log(), reduction="none").mean(dim=[3,4])
    return spectral_convergence_loss.mean(dim=[0,2]), log_mag_stft_loss.mean(dim=[0,2])


def mae_objective(estimation, origin, _):
    return F.l1_loss(estimation, origin, reduction="none").mean(dim=(0,2,3)) # shape: (4)

def mse_objective(estimation, origin, _):
    return F.mse_loss(estimation, origin, reduction="none").mean(dim=(0,2,3)) # shape: (4)

def sdr_objective(estimation, origin, mask=None):
    """
    Scale-invariant signal-to-noise ratio (SI-SNR) loss

    Arguments:
        estimation {torch.tensor} -- separated signal of shape: (B, 4, 1, T)
        origin {torch.tensor} -- ground-truth separated signal of shape (B, 4, 1, T)

    Keyword Arguments:
        mask {torch.tensor, None} -- boolean mask: True when $origin is 0.0; shape (B, 4, 1) (default: {None})

    Returns:
        torch.tensor -- SI-SNR loss of shape: (4)
    """
    if estimation.ndim == 4:
        if estimation.shape[2] > 1:
            return functools.reduce(lambda x,y: x+y, [sdr_objective(estimation[:,:,c:c+1,:], origin[:,:,c:c+1,:], mask=mask[:,:,c:c+1] if mask is not None else None) for c in range(estimation.shape[2])])/estimation.shape[2]
    elif estimation.ndim == 3:
        if estimation.shape[1] > 1:
            return functools.reduce(lambda x,y: x+y, [sdr_objective(estimation[:,c:c+1,:], origin[:,c:c+1,:], mask=mask[:,c:c+1] if mask is not None else None) for c in range(estimation.shape[1])])/estimation.shape[1]
    origin_power = torch.pow(origin, 2).sum(dim=-1, keepdim=True) + 1e-8  # shape: (B, 4, 1, 1)
    scale = torch.sum(origin*estimation, dim=-1, keepdim=True) / origin_power  # shape: (B, 4, 1, 1)

    est_true = scale * origin  # shape: (B, 4, 1, T)
    est_res = estimation - est_true  # shape: (B, 4, 1, T)

    true_power = torch.pow(est_true, 2).sum(dim=-1).clamp(min=1e-8)  # shape: (B, 4, 1)
    res_power = torch.pow(est_res, 2).sum(dim=-1).clamp(min=1e-8)  # shape: (B, 4, 1)

    sdr = 10*(torch.log10(true_power) - torch.log10(res_power))  # shape: (B, 4, 1)

    if mask is not None:
        sdr = (sdr*mask).sum(dim=(0, -1)) / mask.sum(dim=(0, -1)).clamp(min=1e-8)  # shape: (4)
    else:
        sdr = sdr.mean(dim=(0, -1))  # shape: (4)

    return sdr  # shape: (4)


def dissimilarity_loss(latents, mask):
    """
    Minimize the similarity between the different instrument latent representations

    Arguments:
        latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        mask {torch.tensor} -- boolean mask: True when the signal is 0.0; shape (B, 4)

    Returns:
        torch.tensor -- shape: ()
    """
    a_i = (0, 0, 0, 1, 1, 2)
    b_i = (1, 2, 3, 2, 3, 3)

    a = latents[a_i, :, :, :]
    b = latents[b_i, :, :, :]

    count = (mask[:, a_i] * mask[:, b_i]).sum() + 1e-8
    sim = F.cosine_similarity(a.abs(), b.abs(), dim=-1)
    sim = sim.sum(dim=(0, 1)) / count
    return sim.mean()


def similarity_loss(latents, mask):
    """
    Maximize the similarity between the same instrument latent representations

    Arguments:
        latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        mask {torch.tensor} -- boolean mask: True when the signal is 0.0; shape (B, 4)

    Returns:
        torch.tensor -- shape: ()
    """
    a = latents
    b = torch.roll(latents, 1, dims=1)

    count = (mask * torch.roll(mask, 1, dims=0)).sum().clamp(min=1e-8)
    sim = F.cosine_similarity(a, b, dim=-1)
    sim = sim.sum(dim=(0, 1)) / count
    return sim.mean()

def calculate_loss_wo_aux(estimated_separation, true_separation, mask, _, __, ___, args, n_srcs=4):
    """
    The loss function, the sum of 4 different partial losses

    Arguments:
        estimated_separation {torch.tensor} -- separated signal of shape: (B, 4, 1, T)
        true_separation {torch.tensor} -- ground-truth separated signal of shape (B, 4, 1, T)
        mask {torch.tensor} -- boolean mask: True when $true_separation is 0.0; shape (B, 4, 1)
        true_latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        estimated_mix {torch.tensor} -- estimated reconstruction of the mix, shape: (B, 1, T)
        true_mix {torch.tensor} -- ground-truth mixed signal, shape: (B, 1, T)
        args {dict} -- argparse hyperparameters

    Returns:
        (torch.tensor, torch.tensor) -- shape: [(), (7)]
    """
    stats = torch.zeros(n_srcs+3).to(estimated_separation.device)

    if args.loss_criterion == "mse":
        mse = mse_objective(estimated_separation, true_separation, mask)
        stats[:n_srcs] = mse
        total_loss = mse.sum()
    elif args.loss_criterion == "mseall":
        mse = mse_objective(estimated_separation, true_separation, mask)
        input_mse = mse_objective(estimated_separation.sum(dim=1, keepdim=True), true_separation.sum(dim=1, keepdim=True), mask)
        stats[:n_srcs] = mse
        stats[n_srcs] = input_mse
        total_loss = mse.sum() + input_mse.sum()
    elif args.loss_criterion == "mae":
        mae = mae_objective(estimated_separation, true_separation, mask)
        stats[:n_srcs] = mae
        total_loss = mae.sum()
    elif args.loss_criterion == "sisdr":
        sdr = sdr_objective(estimated_separation, true_separation, mask)
        stats[:n_srcs] = sdr
        total_loss = -sdr.sum()
    elif args.loss_criterion == "stft_mse":
        mse = STFT_mse_objective(estimated_separation, true_separation, mask)
        stats[:n_srcs] = mse
        total_loss = mse.sum()
    elif args.loss_criterion == "mae_plus_multiresol_stft":
        mae = mae_objective(estimated_separation, true_separation, mask)
        factor = args.sampling_rate/16000.0
        specconv_loss, logmagstft_loss = multiresolution_STFT_objective(
            estimated_separation, true_separation, 
            n_ffts=[int(_*factor) for _ in [1024, 2048, 512]], 
            hop_lengths=[int(_*factor) for _ in [120, 240, 50]], 
            win_lengths=[int(_*factor) for _ in [600, 1200, 240]], 
            center=True, normalized=False, onesided=True, pad_mode="reflect", eps=1e-8
        )
        sum_loss = mae + (specconv_loss + logmagstft_loss)*0.1
        stats[:n_srcs] = sum_loss
        total_loss = sum_loss.sum()
    else:
        raise NotImplementedError(f'Uknown loss criterion [{args.loss_criterion}]')

    return total_loss, stats

def calculate_loss(estimated_separation, true_separation, mask, true_latents, estimated_mix, true_mix, args):
    """
    The loss function, the sum of 4 different partial losses

    Arguments:
        estimated_separation {torch.tensor} -- separated signal of shape: (B, 4, 1, T)
        true_separation {torch.tensor} -- ground-truth separated signal of shape (B, 4, 1, T)
        mask {torch.tensor} -- boolean mask: True when $true_separation is 0.0; shape (B, 4, 1)
        true_latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        estimated_mix {torch.tensor} -- estimated reconstruction of the mix, shape: (B, 1, T)
        true_mix {torch.tensor} -- ground-truth mixed signal, shape: (B, 1, T)
        args {dict} -- argparse hyperparameters

    Returns:
        (torch.tensor, torch.tensor) -- shape: [(), (7)]
    """
    stats = torch.zeros(7).to(estimated_separation.device)

    if args.loss_criterion == "mse":
        mse = mse_objective(estimated_separation, true_separation, mask)
        stats[:4] = mse
        total_loss = mse.sum()
    elif args.loss_criterion == "mseall":
        mse = mse_objective(estimated_separation, true_separation, mask)
        input_mse = mse_objective(estimated_separation.sum(dim=1, keepdim=True), true_separation.sum(dim=1, keepdim=True), mask)
        stats[:4] = mse
        stats[4] = input_mse
        total_loss = mse.sum() + input_mse.sum()
    elif args.loss_criterion == "mae":
        mae = mae_objective(estimated_separation, true_separation, mask)
        stats[:4] = mae
        total_loss = mae.sum()
    elif args.loss_criterion == "sisdr":
        sdr = sdr_objective(estimated_separation, true_separation, mask)
        stats[:4] = sdr
        total_loss = -sdr.sum()
    elif args.loss_criterion == "stft_mse":
        mse = STFT_mse_objective(estimated_separation, true_separation, mask)
        stats[:4] = mse
        total_loss = mse.sum()
    elif args.loss_criterion == "mae_plus_multiresol_stft":
        mae = mae_objective(estimated_separation, true_separation, mask)
        factor = args.sampling_rate/16000.0
        specconv_loss, logmagstft_loss= multiresolution_STFT_objective(
            estimated_separation, true_separation, 
            n_ffts=[int(_*factor) for _ in [1024, 2048, 512]], 
            hop_lengths=[int(_*factor) for _ in [120, 240, 50]], 
            win_lengths=[int(_*factor) for _ in [600, 1200, 240]], 
            center=True, normalized=False, onesided=True, pad_mode="reflect", eps=1e-8
        )
        sum_loss = mae + (specconv_loss + logmagstft_loss)*0.1
        stats[:4] = sum_loss
        total_loss = sum_loss.sum()
    else:
        raise NotImplementedError(f'Uknown loss criterion [{args.loss_criterion}]')

    reconstruction_sdr = sdr_objective(estimated_mix, true_mix).mean() if args.reconstruction_loss_weight > 0 else 0.0
    stats[4] = reconstruction_sdr
    total_loss += -args.reconstruction_loss_weight * reconstruction_sdr

    if args.similarity_loss_weight > 0.0 or args.dissimilarity_loss_weight > 0.0:
        if mask.shape[-1] > 1:
            collapsed_mask = (mask.sum(2) > 0).float()
            true_latents = true_latents * collapsed_mask.unsqueeze(-1).unsqueeze(-1)
            true_latents = true_latents.transpose(0, 1)
        else:
            mask = mask.squeeze(-1)
            true_latents = true_latents * mask.unsqueeze(-1).unsqueeze(-1)
            true_latents = true_latents.transpose(0, 1)

    dissimilarity = dissimilarity_loss(true_latents, mask) if args.dissimilarity_loss_weight > 0.0 else 0.0
    stats[5] = dissimilarity
    total_loss += args.dissimilarity_loss_weight * dissimilarity

    similarity = similarity_loss(true_latents, mask) if args.similarity_loss_weight > 0.0 else 0.0
    stats[6] = similarity
    total_loss += -args.similarity_loss_weight * similarity

    return total_loss, stats


