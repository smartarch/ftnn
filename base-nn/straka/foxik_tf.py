#!/usr/bin/env python3
import argparse

import h5py
import tensorflow as tf

def load_data(path):
    with h5py.File(path, "r") as data_file:
        train = tf.data.Dataset.from_tensor_slices((data_file["train-inputs"], data_file["train-outputs"]))
        test = tf.data.Dataset.from_tensor_slices((data_file["val-inputs"], data_file["val-outputs"]))
    return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("--activation", default="relu", type=str, help="Activation")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size")
    parser.add_argument("--dev_size", default=0.05, type=float, help="Fraction of train data to use as dev")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs")
    parser.add_argument("--label_smoothing", default=0.2, type=float, help="Label smoothing")
    parser.add_argument("--layer_sizes", default=[256, 256], type=int, nargs="+", help="Hidden layer sizes")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--learning_rate_decay", default=True, action="store_true", help="Decay learning rate")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Use given number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load the data and prepare the datasets
    train, test = load_data(args.input_file)
    train_size = int(len(train) * (1 - args.dev_size))
    train, dev = train.take(train_size), train.skip(train_size)

    train = train.shuffle(10000)
    train, dev, test = train.batch(args.batch_size), dev.batch(args.batch_size), test.batch(args.batch_size)

    # Create the model
    model = tf.keras.models.Sequential()
    for layer_size in args.layer_sizes:
        model.add(tf.keras.layers.Dense(layer_size, activation=getattr(tf.nn, args.activation)))
        model.add(tf.keras.layers.Dropout(args.dropout))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    # Configure model training
    learning_rate = args.learning_rate
    if args.learning_rate_decay:
        learning_rate = tf.keras.experimental.CosineDecay(learning_rate, args.epochs * len(train))
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.BinaryAccuracy("accuracy")],
    )

    # Train and evaluate
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, verbose=2)
    test_logs = model.evaluate(test, return_dict=True, verbose=2)

    print("{} dev:{} test:{}".format(
        " ".join("{}:{}".format(key, value) for key, value in vars(args).items()),
        logs.history["val_accuracy"][-1], test_logs["accuracy"]))
