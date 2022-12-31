import argparse
from pathlib import Path

import librosa
import numpy
import soundfile as sf
import torch
import torch.nn
import torchaudio
from torch import nn
from tqdm import tqdm
from train import define_model
import copy

class DummyTrack:
    def __init__(self, audio) -> None:
        self.audio = audio

class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def sr_specified_inference(self, *args, **kwargs):
        return self.model.sr_specified_inference(*args, **kwargs)
    
    def inference(self, *args, **kwargs):
        return self.model.inference(*args, **kwargs)

    @property
    def base_sr(self):
        return self.model.base_sr
    
def normalize_estimates_by_mse(estimates, mixture, verbose=None):
    '''

    Args:
        estimates (dict[str,numpy.ndarray]): dict of time x ch array
        mixture (numpy.ndarray): time x ch

    '''
    a_l = numpy.array([estimates['drums'][:, 0], estimates['bass'][:, 0], estimates['other'][:, 0], estimates['vocals'][:, 0]]).T # time x inst
    a_r = numpy.array([estimates['drums'][:, 1], estimates['bass'][:, 1], estimates['other'][:, 1], estimates['vocals'][:, 1]]).T # time x inst

    b_l = mixture[:, 0] # time
    b_r = mixture[:, 1] # time

    if verbose: print(a_l.shape, b_l.shape)

    sol_l = numpy.linalg.lstsq(a_l, b_l, rcond=None)[0] # inst
    sol_r = numpy.linalg.lstsq(a_r, b_r, rcond=None)[0] # inst

    e_l = a_l * sol_l
    e_r = a_r * sol_r

    separation = numpy.array([e_l, e_r])  # shape: (channel, time, instrument)

    if verbose: print(separation.shape)

    estimates = {
        'drums': separation[:, :, 0].T,
        'bass': separation[:, :, 1].T,
        'other': separation[:, :, 2].T,
        'vocals': separation[:, :, 3].T,
    }
    return estimates

def resample(audio, input_sr, output_sr, res_type='kaiser_best'):
    '''

    Args:
        audio (numpy.ndarray): ch (optional) x time
        input_sr (int): input sampling rate
        output_sr (int): output sampling rate
    '''
    if input_sr == output_sr:
        return audio
    audio = numpy.asfortranarray(audio)
    return librosa.core.resample(audio, input_sr, output_sr, res_type=res_type, fix=False)
    
def separate_sample(network, track, sample_rate, device, verbose=False):
    audio = track.audio.copy().astype('float32').transpose(1, 0) # ch x time

    mix = [audio]
    mix = [librosa.util.fix_length(m, size=(mix[0].shape[-1]+1)*(2**i)) for i, m in enumerate(mix)]
    mix = [torch.from_numpy(s).float().to(device).unsqueeze_(1) for s in mix]
    mix = [s / s.std(dim=-1, keepdim=True) for s in mix]

    mix_left = [s[0:1, :, :] for s in mix]
    mix_right = [s[1:2, :, :] for s in mix]
    del mix

    network.eval()
    with torch.inference_mode():
        separation_left = network.sr_specified_inference(mix_left, sample_rate=sample_rate, n_chunks=8)[-1].cpu().squeeze_(2)  # shape: (4, T)
        separation_right = network.sr_specified_inference(mix_right, sample_rate=sample_rate, n_chunks=8)[-1].cpu().squeeze_(2)  # shape: (4, T)

        separation = torch.cat([separation_left, separation_right], 0) # 2 x 4 x time
        separation = separation.numpy()

    if verbose: print(separation.shape)

    estimates = {
        'drums': separation[:, 0, :track.audio.shape[0]].T, # time x ch
        'bass': separation[:, 1, :track.audio.shape[0]].T, # time x ch
        'other': separation[:, 2, :track.audio.shape[0]].T, # time x ch
        'vocals': separation[:, 3, :track.audio.shape[0]].T, # time x ch
    }

    estimates = normalize_estimates_by_mse(estimates, track.audio, verbose=verbose)

    return estimates

def separate_sample_with_signal_resampling(network, track, sample_rate, device, verbose=False):
    audio = track.audio.copy().astype('float32').transpose(1, 0) # ch x time
    with torch.inference_mode():
        audio = torchaudio.functional.resample(torch.tensor(audio), sample_rate, network.base_sr).cpu().numpy()
    track_audio = audio.copy().transpose(1,0) # time x ch, mixture of trained sampling frequency

    mix = [audio]
    mix = [librosa.util.fix_length(m, size=(mix[0].shape[-1]+1)*(2**i)) for i, m in enumerate(mix)]
    mix = [torch.from_numpy(s).float().to(device).unsqueeze_(1) for s in mix]
    mix = [s / s.std(dim=-1, keepdim=True) for s in mix]

    mix_left = [s[0:1, :, :] for s in mix]
    mix_right = [s[1:2, :, :] for s in mix]
    del mix

    network.eval()
    with torch.inference_mode():
        separation_left = network.inference(mix_left, n_chunks=8)[-1].cpu().squeeze_(2)  # shape: (4, T)
        separation_right = network.inference(mix_right, n_chunks=8)[-1].cpu().squeeze_(2)  # shape: (4, T)
        separation = torch.cat([separation_left, separation_right], 0) # 2 x 4 x time
        separation = separation.numpy() # 2 x 4 x time 

    if verbose: print(separation.shape)

    estimates = {
        'drums': separation[:, 0, :track_audio.shape[0]].T, # time x ch
        'bass': separation[:, 1, :track_audio.shape[0]].T, # time x ch
        'other': separation[:, 2, :track_audio.shape[0]].T, # time x ch
        'vocals': separation[:, 3, :track_audio.shape[0]].T, # time x ch
    }

    estimates = normalize_estimates_by_mse(estimates, track_audio, verbose=verbose)

    for k in ["drums", "bass", "other", "vocals"]:
        estimates[k] = torchaudio.functional.resample(torch.tensor(estimates[k].T), network.base_sr, sample_rate).cpu().numpy()[:,:track.audio.shape[0]].T
    return estimates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Path to the trained model.")
    parser.add_argument("--input_files", required=True, type=str, nargs="*", help="Audio file paths.")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory.")
    parser.add_argument("--use_signal_resampling", action="store_true", help="Set this option when you use signal resampling.")
    parser.add_argument("--res_type", choices=["kaiser_best", "kaiser_fast"], default="kaiser_best", help="Signal resampling type")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    checkpoint = torch.load(args.model)
    model_args = copy.deepcopy(checkpoint["args"])

    # Define model
    print(f'Loading {args.model}', flush=True)
    network = define_model(device, model_args)            
    network.load_state_dict(checkpoint["state_dict"])
    assert model_args.stages_num == 1

    network = InferenceWrapper(network)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_signal_resampling:
        print("Use signal resampling (i.g., not SFI)", flush=True)

    with tqdm([Path(_) for _ in args.input_files], desc="Separating mixtures") as pbar:
        for input_audio_filename in pbar:
            audio_data, sample_rate = sf.read(input_audio_filename)
            pbar.set_description_str(f'{input_audio_filename} (sample rate: {sample_rate})')

            track = DummyTrack(audio_data)
            if args.use_signal_resampling:
                estimates = separate_sample_with_signal_resampling(network, track, sample_rate, device=device, verbose=None)
            else:
                estimates = separate_sample(network, track, sample_rate, device=device, verbose=None)
            for name, waveform in estimates.items():
                # out_filename = output_dir / f'{input_audio_filename.stem}' / f'{name}.wav'
                out_filename = output_dir / f'{sample_rate:05d}' / f'{input_audio_filename.parent.stem}' / f'{name}.wav'
                out_filename.parent.mkdir(parents=True, exist_ok=True)
                waveform = waveform.astype('f').copy()
                sf.write(out_filename, waveform, samplerate=sample_rate)
