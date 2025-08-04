import shutil
import os
import numpy
import json
import random
import argparse
from tqdm import tqdm


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser(
        description='split event frames into train test and validation')
    parser.add_argument(
        "--frames_dir", required=True, type=str, 
        help="directory with blender event frames")

    args = parser.parse_args()

    frames_dir = args.frames_dir

    file_names = os.listdir(os.path.join(frames_dir))
    random.shuffle(file_names)

    makedir(os.path.join(frames_dir, 'train'))
    makedir(os.path.join(frames_dir, 'validation'))
    makedir(os.path.join(frames_dir, 'test'))

    total = len(file_names)
    train_split_size = int(total * 0.7)
    val_split_size = int(total * 0.15)
    split1_start = 0
    split2_start = split1_start + train_split_size
    split3_start = split2_start + val_split_size

    for file_name in tqdm(file_names[0:split2_start]):
        file_path = os.path.join(frames_dir, file_name)
        shutil.copy(file_path, os.path.join(frames_dir, 'train'))

    for file_name in tqdm(file_names[split2_start:split3_start]):
        file_path = os.path.join(frames_dir, file_name)
        shutil.copy(file_path, os.path.join(frames_dir, 'validation'))

    for file_name in tqdm(file_names[split3_start:]):
        file_path = os.path.join(frames_dir, file_name)
        shutil.copy(file_path, os.path.join(frames_dir, 'test'))

if __name__ == "__main__":
    main()