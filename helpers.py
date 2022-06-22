import random
import torch
import pandas as pd

from typing import Tuple, Dict, Any


def get_data(
    full: bool,
    seed: int,
    test_size: float
) -> Tuple[Tuple[str], torch.Tensor, Tuple[str], torch.Tensor]:
    """
    Loads training and testing data

    :param full: whether to use the full dataset (True) or not (False)
    :param seed: a seed to determine the shuffling
    :param test_size: the relative size of the test set
    :return: a tuple (X_train, y_train, X_test, y_test)
    """
    prefix = 'twitter-datasets/train_'
    suffix = '_full' if full else ''
    with open(f'{prefix}neg{suffix}.txt', 'r') as f:
        neg = f.read().split('\n')[:-1]
    with open(f'{prefix}pos{suffix}.txt', 'r') as f:
        pos = f.read().split('\n')[:-1]

    data = list(zip(neg, [-1] * len(neg))) + list(zip(pos, [1] * len(pos)))
    random.Random(seed).shuffle(data)

    split_idx = int((1 - test_size) * len(data))
    train = data[:split_idx]
    test = data[split_idx:]

    X_train, y_train = list(zip(*train))
    X_test, y_test = list(zip(*test))
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    return X_train, y_train, X_test, y_test


def calc_clf_acc(
    true: torch.Tensor,
    predicted: torch.Tensor
) -> float:
    """
    Calculates the classification accuracy.

    :param true: tensor of true labels (-1 for neg, 1 for pos)
    :param predicted: tensor of predicted labels (-1 for neg, 1 for pos)
    :return: the classification accuracy (n_correct / n_total)
    """
    return (true == predicted).float().mean().item()


def predict_holdout(
    clf: Any,
    out_path: str
) -> None:
    """
    Predicts on the holdout "test_data.txt" using provided classifier.

    :param clf: Classifier with .predict() method
    :param out_path: where to save output to
    """
    with open('twitter-datasets/test_data.txt', 'r') as f:
        X = [','.join(l.split(',')[1:]) for l in f.read().split('\n')[:-1]]

    ids = range(1, len(X) + 1)
    predictions = clf.predict(X).to(int).tolist()
    submission = pd.DataFrame([ids, predictions], index=['Id', 'Prediction']).T

    submission.to_csv(out_path, index=False)


def pipeline(
    full: bool,
    seed: int,
    test_size: float,
    model_class: Any,
    model_args: Dict[str, Any],
    out_path: str=None
) -> None:
    """
    Loads data, fits and evaluates the model and predicts on holdout set.

    :param full: whether to use the full dataset (True) or not (False)
    :param seed: a seed to determine the shuffling
    :param test_size: the relative size of the test set
    :param model_class: a class implementing .fit() and .predict() methods
    :param model_args: a dictionary of model arguments

    """
    X_train, y_train, X_test, y_test = get_data(full, seed, test_size)

    clf = model_class(model_args)
    clf.fit(X_train, y_train)
    
    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)

    train_acc = calc_clf_acc(y_train, y_train_hat)
    test_acc = calc_clf_acc(y_test, y_test_hat)
    print(f'Training accuracy: {train_acc * 100:.2f}%')
    print(f'Testing accuracy: {test_acc * 100:.2f}%')

    if out_path is not None:
        predict_holdout(clf, out_path)
