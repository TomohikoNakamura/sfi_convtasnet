import argparse
from pathlib import Path

import librosa
import musdb
import museval
import numpy
import pandas as pd
import simplejson
import soundfile as sf
import torch
import torch.nn
import torchaudio
from pandas import json_normalize
from torch import nn
from tqdm import tqdm
from train import define_model

from separate_audiofile import InferenceWrapper, separate_sample, separate_sample_with_signal_resampling 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Directory of the model to evaluate.")
    parser.add_argument("--musdb_path", required=True, type=str, help="Path to the MUSDB18 dataset.")
    parser.add_argument("--sample_rate", required=True, type=int, help="Sampling rate")
    parser.add_argument("--save_waveforms", action="store_true", help="If true, save estimated waveforms.")
    parser.add_argument("--use_signal_resampling", action="store_true")
    parser.add_argument("--res_type", choices=["kaiser_best", "kaiser_fast"], default="kaiser_best")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    checkpoint = torch.load(f"{args.model_dir}/best_checkpoint")
    # model_args = checkpoint["args"]
    import copy
    model_args = copy.deepcopy(checkpoint["args"])
    model_args["model_type"] = "sfi_convtasnet"

    # Define model
    network = define_model(device, model_args)            
    network.load_state_dict(checkpoint["state_dict"])
    assert model_args.stages_num == 1

    network = InferenceWrapper(network)

    mus_test = musdb.DB(root=f'{args.musdb_path}_{args.sample_rate}', subsets="test", is_wav=True)
    assert len(mus_test.tracks) > 0, f'target_dir={args.musdb_path}_{args.sample_rate}'

    model_dir = Path(args.model_dir)
    if args.use_signal_resampling:
        output_dir_head = f'{model_dir}/test{args.sample_rate:d}_sigresample'
    else:
        output_dir_head = f'{model_dir}/test{args.sample_rate:d}'

    output_dir = f"{output_dir_head}/scores"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # print("separating...")
    def evaluate(track_estimates):
        track, estimates = track_estimates
        if args.save_waveforms:
            for name, waveform in estimates.items():
                out_filename = Path(output_dir_head) / 'test' / track.name / f'{name}.wav'
                out_filename.parent.mkdir(parents=True, exist_ok=True)
                waveform = waveform.astype('f').copy()
                sf.write(out_filename, waveform, samplerate=args.sample_rate)
        museval.eval_mus_track(track, estimates, output_dir=output_dir)

    tasks = []
    for i, track in enumerate(tqdm(mus_test.tracks, desc="Evaluate")):
        if args.use_signal_resampling:
            estimates = separate_sample_with_signal_resampling(network, track, args.sample_rate, device=device, verbose=None)
        else:
            estimates = separate_sample(network, track, args.sample_rate, device=device, verbose=None)
        evaluate((track, estimates))

    print("Everything is evaluated")

    def json2df(json_string, track_name):
        df = json_normalize(json_string['targets'], ['frames'], ['name'])
        df.columns = [col.replace('metrics.', '') for col in df.columns]
        df = pd.melt(
            df,
            var_name='metric',
            value_name='score',
            id_vars=['time', 'name'],
            value_vars=['SDR', 'SAR', 'ISR', 'SIR']
        )
        df['track'] = track_name
        df = df.rename(index=str, columns={"name": "target"})
        return df

    scores = museval.EvalStore(frames_agg='median')
    p = Path(output_dir)
    json_paths = p.glob('test/**/*.json')
    for json_path in json_paths:
        with open(json_path) as json_file:
            json_string = simplejson.loads(json_file.read())
        track_df = json2df(json_string, json_path.stem)
        scores.add_track(track_df)

    print(f"### Target sampling rate = {args.sample_rate}")
    print(scores)
