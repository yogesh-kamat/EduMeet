import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
# GPU config if using gpu then uncomment following lines
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

from input_pipeline import get_dataset


class_names = ['Boredom', 'Engagement', 'Confusion', 'Frustration']


def print_accuracy(result):
    """Print accuracy for all the classes."""
    print("Accuracy: ")
    for i, acc in enumerate(result[5:]):
        print(f"{class_names[i]}: {acc*100}")


def test(dataset, model, model_fine, model_fc, model_fc_fine, out_dir):
    """
    Evaluate test data on all the models.

    Args:
        dataset: tf.data.Dataset object as from input_pipeline
        model: Model instance with no finetune and no Dense layers.
        model_fine: Model instance with finetune and no Denselayers.
        model_fc: Model instance with no finetune and Dense layers.
        model_fc_fine: Model instance with finetune and Dense layers.
        out_dir: Directory to store results of evaluation.
    """
    if out_dir.parts[-1] == "fps_0.7":
        model_result = model.evaluate(dataset)
        np.save(f"{str(out_dir)}/no_fc_no_finetune", np.array(model_result))
        model_fine_result = model_fine.evaluate(dataset)
        np.save(f"{str(out_dir)}/no_fc_finetune", np.array(model_fine_result))
        model_fc_result = model_fc.evaluate(dataset)
        np.save(f"{str(out_dir)}/fc_no_finetune", np.array(model_fc_result))
        model_fc_fine_result = model_fc_fine.evaluate(dataset)
        np.save(f"{str(out_dir)}/fc_finetune", np.array(model_fc_fine_result))
        print("=" * 50)
        print("=" * 50)
        print("Testing on data with frame rate as 0.7fps")
        print("1. model with no fc layers and no finetunning")
        print_accuracy(model_result)
        print("=" * 50)
        print("2. model with no fc layers and finetunning")
        print_accuracy(model_fine_result)
        print("=" * 50)
        print("3. model with fc layers and no finetunning")
        print_accuracy(model_fc_result)
        print("=" * 50)
        print("4. model with fc layers and finetunning")
        print_accuracy(model_fc_fine_result)
        print("=" * 50)
        print("=" * 50)
    else:
        model_fc_result = model_fc.evaluate(dataset)
        np.save(f"{str(out_dir)}/fc_no_finetune", np.array(model_fc_result))
        model_fc_fine_result = model_fc_fine.evaluate(dataset)
        np.save(f"{str(out_dir)}/fc_finetune", np.array(model_fc_fine_result))
        print("=" * 50)
        print("=" * 50)
        print("Testing on data with default frame rate")
        print("1. model with fc layers and no finetunning")
        print_accuracy(model_fc_result)
        print("=" * 50)
        print("2. model with fc layers and finetunning")
        print_accuracy(model_fc_fine_result)
        print("=" * 50)
        print("=" * 50)


def main(model_dir, numpy_dir, out_dir):
    """
    Create neccessary directories, read the trained model and call
    test to evaluate.

    Args:
        model_dir: Directory which contains trained model.
        numpy_dir: Dirctory which contains filepath, label array.
        out_dir: Directory to store results of evaluation as array.
    """
    out_dir = Path(out_dir)
    odir1 = out_dir / "fps_0.7"
    odir1.mkdir(parents=True, exist_ok=True)
    odir2 = out_dir / "fps_default"
    odir2.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model = load_model(f"{str(model_dir)}/Xception_on_DAiSEE.h5")
    model_fine = load_model(f"{str(model_dir)}/Xception_on_DAiSEE_finetune.h5")
    model_fc = load_model(f"{str(model_dir)}/Xception_on_DAiSEE_fc.h5")
    model_fc_fine = load_model(
        f"{str(model_dir)}/Xception_on_DAiSEE_finetune_fc.h5")

    test_ds = get_dataset("Test", numpy_dir)
    test_ds_final = get_dataset("FinalTest", numpy_dir)

    test(test_ds, model, model_fine, model_fc, model_fc_fine, odir1)
    test(test_ds_final, model, model_fine, model_fc, model_fc_fine, odir2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model')
    grp = parser.add_argument_group('Required arguments')
    grp.add_argument("-i", "--model_dir", type=str, required=True,
                     help="Directory which contains models.")
    grp.add_argument("-n", "--numpy_dir", type=str, required=True,
                     help="Directory which contains filepath and label array.")
    grp.add_argument("-o", "--out_dir", type=str, required=True,
                     help="Directory to store result.")
    args = parser.parse_args()
    main(args.model_dir, args.numpy_dir, args.out_dir)
