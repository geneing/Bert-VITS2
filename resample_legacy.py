import os
import argparse
import librosa
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm

from config import config


def process(item):
    wav_path, out_path, args = item
    
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
        wav, sr = librosa.load(wav_path, sr=args.sr)
        basedir = os.path.dirname(out_path)
        if not os.path.exists(basedir):
            os.makedirs(basedir, exist_ok=True)
        print(f"{wav_path=} {out_path=}\n")
        soundfile.write(out_path, wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="cpu processes",
    )
    args, _ = parser.parse_known_args()
    # autodl no-card mode will recognize 46 CPUs
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
    pool = Pool(processes=processes)

    tasks = []

    for dirpath, _, filenames in os.walk(args.in_dir):
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                inpath = os.path.join(dirpath, filename)
                outpath = inpath.replace(args.in_dir, args.out_dir)
                tasks.append((inpath, outpath, args))
    
    for _ in tqdm(
        pool.imap_unordered(process, tasks),
    ):
        pass

    pool.close()
    pool.join()

    print("Audio resampling complete!")
