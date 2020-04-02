import argparse
from pathlib import Path

from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# GPU config if using gpu then uncomment following lines
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

from input_pipeline import get_dataset, img_width, img_height


finetune_at = 116
base_learning_rate = 0.0001
old_epoch = 0
epochs = 10


def get_model(weight_dir, out_dir, fullyconnected=False, finetune=False):
    """
    Creates custom model on top of Xception for user engagement
    recognition.

    Returns:
        model for the training.
    """
    weights = Path(weight_dir)
    if finetune:
        if fullyconnected:
            base_model = load_model(str(out_dir) + "/Xception_on_DAiSEE_fc.h5")
        else:
            base_model = load_model(str(out_dir) + "/Xception_on_DAiSEE.h5")

        base_model.trainable = True
        for layer in base_model.layers[:finetune_at]:
            layer.trainable = False
        return base_model
    else:
        xception = "xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
        weights = weights / xception
        base_model = Xception(weights=str(weights),
                              include_top=False,
                              input_shape=(img_width, img_height, 3))

        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        if fullyconnected:
            x = Dense(128, activation="relu", name="fc1")(x)
            x = Dense(64, activation="relu", name="fc2")(x)
        boredom = Dense(4, name="y1")(x)
        engagement = Dense(4, name="y2")(x)
        confusion = Dense(4, name="y3")(x)
        frustration = Dense(4, name="y4")(x)
        model = Model(inputs=base_model.input,
                      outputs=[boredom, engagement, confusion, frustration])
    return model


def main(weight_dir, numpy_dir, out_dir, fullyconnected=False, finetune=False):
    """
    Creates trained model for user engagement recognition.

    Args:
        weight_dir: Directory which contains pretrained weights for
            Xception with include_top=False.
        numpy_dir: Directory which contains filepath and labels array.
        out_dir: Directory to store models and logs.
        fullyconnected: Boolean indicating wheater to add two fully
            connected layers on top of Xception before the
            classification head or not, default to False.
        finetune: Boolean indicating wheater to finetune the model or
            not, default to False.
    """
    global old_epoch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds = get_dataset("Train", numpy_dir)
    validation_ds = get_dataset("Validation", numpy_dir)
    model = get_model(weight_dir, out_dir, fullyconnected, finetune)

    if finetune:
        lr = base_learning_rate / 10
        finetune_epochs = 10
        if fullyconnected:
            model_path = str(out_dir) + "/Xception_on_DAiSEE_finetune_fc.h5"
        else:
            model_path = str(out_dir) + "/Xception_on_DAiSEE_finetune.h5"
    else:
        lr = base_learning_rate
        finetune_epochs = 0
        if fullyconnected:
            model_path = str(out_dir) + "/Xception_on_DAiSEE_fc.h5"
        else:
            model_path = str(out_dir) + "/Xception_on_DAiSEE.h5"

    model.compile(optimizer=RMSprop(learning_rate=lr),
                  loss={"y1": SparseCategoricalCrossentropy(from_logits=True),
                        "y2": SparseCategoricalCrossentropy(from_logits=True),
                        "y3": SparseCategoricalCrossentropy(from_logits=True),
                        "y4": SparseCategoricalCrossentropy(from_logits=True)},
                  metrics={"y1": "sparse_categorical_accuracy",
                           "y2": "sparse_categorical_accuracy",
                           "y3": "sparse_categorical_accuracy",
                           "y4": "sparse_categorical_accuracy"})
    print(model.summary())

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-2,
                      patience=2, verbose=1),
        TensorBoard(log_dir=str(log_dir))
    ]

    total_epochs = epochs + finetune_epochs
    history = model.fit(train_ds,
                        epochs=total_epochs,
                        initial_epoch=old_epoch,
                        callbacks=callbacks,
                        validation_data=validation_ds)
    if finetune:
        old_epoch = 0
    else:
        old_epoch = history.epoch[-1]
    print(history.history)
    model.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get trained model.")
    grp = parser.add_argument_group("Required arguments")
    grp.add_argument("-i", "--weight_dir", type=str, required=True,
                     help="Directory which contains pretrained weights for "
                     "Xception.")
    grp.add_argument("-n", "--numpy_dir", type=str, required=True,
                     help="Directory which contains filepath and label array.")
    grp.add_argument("-o", "--out_dir", type=str, required=True,
                     help="Directory to store trained model and logs.")
    args = parser.parse_args()
    main(args.weight_dir, args.numpy_dir, args.out_dir)
    main(args.weight_dir, args.numpy_dir, args.out_dir, finetune=True)
    main(args.weight_dir, args.numpy_dir, args.out_dir, fullyconnected=True)
    main(args.weight_dir, args.numpy_dir, args.out_dir,
         fullyconnected=True, finetune=True)
