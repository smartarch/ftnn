#!/usr/bin/env python3
import argparse

import h5py
import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline

def load_data(path):
    with h5py.File(path, "r") as data_file:
        x_train, y_train = np.array(data_file["train-inputs"]), np.array(data_file["train-outputs"]).ravel()
        x_test, y_test = np.array(data_file["val-inputs"]), np.array(data_file["val-outputs"]).ravel()
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("--dev_size", default=0.05, type=float, help="Fraction of train data to use as dev")
    parser.add_argument("--max_depth", default=5, type=int, help="Max GBDT depth")
    parser.add_argument("--model", default="gbdt", type=str, help="Model architecture")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--subsample", default=0.5, type=float, help="Subsample data during training")
    parser.add_argument("--trees", default=200, type=int, help="GBDT Trees")
    args = parser.parse_args()

    # Load the data and prepare the datasets
    (x_train, y_train), (x_test, y_test) = load_data(args.input_file)
    x_train, x_dev, y_train, y_dev = sklearn.model_selection.train_test_split(
        x_train, y_train, test_size=args.dev_size, random_state=args.seed, stratify=y_train)

    # Create the model
    if args.model == "gbdt":
        model = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=args.trees, subsample=args.subsample, max_depth=args.max_depth, verbose=1)
    else:
        raise ValueError("Unknown model {}".format(args.model))

    # Train and evaluate
    model.fit(x_train, y_train)
    dev_accuracy = model.score(x_dev, y_dev)
    test_accuracy = model.score(x_test, y_test)

    print("{} dev:{} test:{}".format(
        " ".join("{}:{}".format(key, value) for key, value in vars(args).items()), dev_accuracy, test_accuracy))
