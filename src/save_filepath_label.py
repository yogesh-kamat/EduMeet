import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from progressbar import ProgressBar


def save_filepath_label(usage, frame_dir, label_dir, out_dir):
    """
    Save filepaths of all the frames and their respective output label
    as numpy array.
    Namely this numpy array's would be x_train, y_train, etc.
    This numpy array would be directly used in input pipeline for
    training and testing of model.

    Args:
        usage: Specify it as Train, Test, Validation or FinalTest.
        frame_dir: Path to directory which contains extracted frames.
        label_dir: Path to directory which contains labels csv files.
        out_dir: Path to directory which will be used to store numpy
            arrays of filepath and labels.
    """
    frame_dir = frame_dir / usage

    if usage == "FinalTest":
        label_path = str(label_dir) + "/TestLabels.csv"
    else:
        label_path = str(label_dir) + f"/{usage}Labels.csv"

    labeldf = pd.read_csv(label_path)
    nrows = len(list(frame_dir.glob("*.jpeg")))
    ncols = len(labeldf.columns) - 1
    filepath = np.empty((nrows,), dtype=np.object)
    label = np.empty((nrows, ncols), dtype=np.float32)
    print(f"Getting filepath and labels for {usage}")
    with ProgressBar(max_value=nrows) as bar:
        for i, frame in enumerate(frame_dir.glob("*.jpeg")):
            filepath[i] = str(frame)
            framename = frame.parts[-1]
            frameid = framename[:framename.find("_")]
            video = frameid + ".avi"
            if labeldf['ClipID'].str.contains(video).any():
                lidx = labeldf.index[labeldf['ClipID'].str.contains(video)]
            else:
                video = frameid + ".mp4"
                lidx = labeldf.index[labeldf['ClipID'].str.contains(video)]
            label[i] = labeldf.iloc[lidx, 1:]
            bar.update(i)

    np.random.seed(100)
    indices = np.random.permutation(nrows)
    filepath = filepath[indices]
    label = label[indices]
    np.save(f"{str(out_dir)}/x_{usage.lower()}", filepath, allow_pickle=True)
    np.save(f"{str(out_dir)}/y_{usage.lower()}", label)
    return filepath, label


def main(frame_dir, label_dir, out_dir):
    """
    Call the savefile_path_label function for Train, Test, Validatin and
    FinalTest.
    """
    frame_dir = Path(frame_dir)
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_filepath_label("Train", frame_dir, label_dir, out_dir)
    save_filepath_label("Test", frame_dir, label_dir, out_dir)
    save_filepath_label("Validation", frame_dir, label_dir, out_dir)
    save_filepath_label("FinalTest", frame_dir, label_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save filepath and labels.")
    grp = parser.add_argument_group("Required arguments")
    grp.add_argument("-i", "--frame_dir", type=str, required=True,
                     help="Directory which contains all frames.")
    grp.add_argument("-l", "--label_dir", type=str, required=True,
                     help="Directory which contains labels csv.")
    grp.add_argument("-o", "--out_dir", type=str, required=True,
                     help="Directory to store frames and labels.")
    args = parser.parse_args()
    main(args.frame_dir, args.label_dir, args.out_dir)
