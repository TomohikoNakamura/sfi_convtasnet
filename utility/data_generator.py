import musdb
import librosa
import numpy as np
import os
import argparse

from concurrent import futures
from tqdm import tqdm

def _process(track_i, track, target_sr, save_only_mono, directory):
    original_sr = track.rate

    mix = librosa.core.resample(track.audio.T, original_sr, target_sr)
    drums = librosa.core.resample(track.targets['drums'].audio.T, original_sr, target_sr)
    bass = librosa.core.resample(track.targets['bass'].audio.T, original_sr, target_sr)
    other = librosa.core.resample(track.targets['other'].audio.T, original_sr, target_sr)
    vocal = librosa.core.resample(track.targets['vocals'].audio.T, original_sr, target_sr)
    acc = librosa.core.resample(track.targets['accompaniment'].audio.T, original_sr, target_sr)

    stereo = [mix, drums, bass, other, vocal, acc]
    length = min([t.shape[1] for t in stereo])
    if length <= 1:
        return

    left = np.array([t[0, :length] for t in stereo])
    right = np.array([t[1, :length] for t in stereo])
    mono = np.array([librosa.to_mono(t[:, :length]) for t in stereo])

    if save_only_mono:
        together = mono
    else:
        together = np.array([left, right, mono])

    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed(f'{directory}/{track_i:04d}', together.astype('float32'))

def multi_process(directory, sources, target_sr, save_only_mono=False, n_threads=1):
    target_sr = int(target_sr)
    with futures.ProcessPoolExecutor(n_threads) as pool:
        tasks = []
        for track_i, track in enumerate(sources):
            tasks.append(pool.submit(_process, track_i, track, target_sr, save_only_mono, directory))
            print(f"Track: {track_i}, sampling rate: {target_sr}")
        pbar = tqdm(tasks)
        for t in pbar:
            t.result()
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb_path", required=True, type=str, help="Path to the MUSDB18 dataset.")
    parser.add_argument("--setup_file", type=str, default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n_threads", type=int, default=12)
    parser.add_argument("--is_wav", action="store_true")
    args = parser.parse_args()

    mus_train = musdb.DB(root=args.musdb_path, subsets="train", split="train", setup_file=args.setup_file, is_wav=args.is_wav)
    mus_val = musdb.DB(root=args.musdb_path, subsets="train", split="valid", setup_file=args.setup_file, is_wav=args.is_wav)

    print(f"The training set size: {len(mus_train)}")
    print(f"The validation set size: {len(mus_val)}\n")

    for sample_rate in [32]:
        print("Converting the training set...")
        multi_process(f"{args.outdir}/train_{sample_rate}", mus_train, sample_rate*1000, save_only_mono=False, n_threads=args.n_threads)
        print("converting the validation set...")
        multi_process(f"{args.outdir}/validation_{sample_rate}", mus_val, sample_rate*1000, save_only_mono=False, n_threads=args.n_threads)
        print()
