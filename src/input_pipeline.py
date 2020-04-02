import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class_names = np.array(
    ['Boredom', 'Engagement', 'Confusion', 'Frustration']
)
autotune = tf.data.experimental.AUTOTUNE
img_width = 299
img_height = 299
batch_size = 16
shuffle_buffer_size = 2000


def show_batch(image, label):
    """
    Show batch of images and labels using matplotlib.

    Args:
        image: Batch of images.
        label: Batch of labels for the given batch of images.
    """
    image = image.numpy()
    plt.figure(figsize=(15, 15))
    for i in range(batch_size):
        plt.subplot(6, 6, i + 1)
        imgtitle = [label["y1"][i].numpy().item(),
                    label["y2"][i].numpy().item(),
                    label["y3"][i].numpy().item(),
                    label["y4"][i].numpy().item()]
        plt.imshow(np.uint8(image[i] * 255))
        plt.title(imgtitle, fontsize=8)
        plt.axis('off')
    plt.show()


def parse_function(filepath, label):
    """
    Read the images from given file path and do final preprocessing
    on the images and labels.

    Args:
        filepath: Path where the image is stored.
        label: Output label for the given image.

    Returns:
        Two tf.Tensor objects which contains transformed image and label.
    """
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(contents=image, channels=3)
    image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
    image = tf.image.resize(images=image,
                            size=[img_width, img_height],
                            method=tf.image.ResizeMethod.BILINEAR,
                            antialias=True)
    return image, label


def get_dataset(usage, numpy_dir):
    """
    Create tf.data.Dataset object for the input pipeline from filepath
    and labels numpy arrays.
    This input pipeline will be used to train and test the model.

    Args:
        usage: Specify it as either Train, Test, or Validation or
            FinalTest.
        numpy_dir: Path to directory which contains numpy arrays of
            filepath and labels.

    Returns:
        tf.data.Dataset object.
    """
    numpy_dir = Path(numpy_dir)
    x = np.load(numpy_dir / f'x_{usage.lower()}.npy', allow_pickle=True)
    y = np.load(numpy_dir / f'y_{usage.lower()}.npy')
    dataset = tf.data.Dataset.from_tensor_slices(
        (x, {"y1": y[:, :1], "y2": y[:, 1:2],
             "y3": y[:, 2:3], "y4": y[:, :3:4]})
    )
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=autotune)
    if usage == 'Train':
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size,
                                  reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(autotune)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Input pipeline for model training and testing.')
    grp = parser.add_argument_group('Required arguments')
    grp.add_argument("-i", "--numpy_dir", type=str, required=True,
                     help="Directory which contains filepath and label array.")
    grp.add_argument('-s', '--subdir', type=str, required=True,
                     help='Specify as Train, Test or Validation')
    args = parser.parse_args()
    ds = get_dataset(args.subdir, args.numpy_dir)
    image_batch, label_batch = next(iter(ds))
    show_batch(image_batch, label_batch)
