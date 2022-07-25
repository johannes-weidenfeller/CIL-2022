import torch
import pandas as pd
import numpy as np
import random

from typing import Tuple, Iterable, Callable


def interleave_uniformly(
    a: Iterable,
    b: Iterable,
    reverse_b: bool=True
) -> Iterable:
    """
    interleave two iterables such ratio a / b is roughly preserved throughout any interval

    :param a: an iterable
    :param b: an iterable
    :param reverse_b: whether or not to reverse the order of b
    :return: a and b interleaved
    """
    if reverse_b:
        b = b[::-1]
    n = len(a) + len(b)
    true_ratio = len(a) / n
    c = []
    a_count, b_count = 0, 0
    running_ratio = 0
    for i in range(n):
        if running_ratio < true_ratio:
            c.append(a[a_count])
            a_count += 1
        else:
            c.append(b[b_count])
            b_count += 1
        running_ratio = a_count / (a_count + b_count)
    return c


def shuffle_slightly(
    a: Iterable,
    shuffle_factor: float
) -> Iterable:
    """
    shuffles a list slightly by swapping (shuffle_factor * 100)% random pairs of elements

    :param a: list
    :param shuffle_factor: a percentage, in range [0, 1]
    """
    n = len(a)
    n_shuffles = int(n * shuffle_factor)
    for i in range(n_shuffles):
        j = random.randint(0, n - 1)
        k = random.randint(0, n - 1)
        a[j], a[k] = a[k], a[j]
    return a


def get_mispredictions(
    X: Tuple[str],
    y: torch.Tensor,
    preds: torch.Tensor
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    returns the subset of tweets which were mispredicted

    :param X: tweets
    :param y: labels
    :param preds: predictions
    """
    mispred_idx = (y != preds).tolist()
    X_mispredicted = tuple(pd.Series(X)[mispred_idx].tolist())
    y_mispredicted = y[mispred_idx]
    return X_mispredicted, y_mispredicted


def get_correct_predictions(
    X: Tuple[str],
    y: torch.Tensor,
    preds: torch.Tensor
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    returns the subset of tweets which were predicted correctly

    :param X: tweets
    :param y: labels
    :param preds: predictions
    """
    match_idx = (y == preds).tolist()
    X_correct = tuple(pd.Series(X)[match_idx].tolist())
    y_correct = y[match_idx]
    return X_correct, y_correct


def get_model_confidences(
    probs: torch.Tensor
) -> torch.Tensor:
    """
    returns the confidence of the model

    :param probs: probabilities
    """
    pos_probs = probs.numpy()[:, 1]
    confidences = 2 * np.abs(probs - 0.5)
    return confidences


def get_errors(
    y: torch.Tensor,
    probs: torch.Tensor
) -> torch.Tensor:
    """
    returns the prediction error
    
    :param y: labels
    :param probs: probabilities
    """
    labels = y / 2 + 0.5
    pos_probs = probs.numpy()[:, 1]
    errors = np.abs(labels - pos_probs)
    return errors


def get_highest(
    X: Tuple[str],
    y: torch.Tensor,
    scores: torch.Tensor,
    pct: float
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    returns the subset of pct% tweets with highest scores
    """
    threshold = np.quantile(scores.numpy(), 1 - pct)
    top_idx = (scores > threshold).tolist()
    X_top = tuple(pd.Series(X)[top_idx])
    y_top = y[top_idx]
    return X_top, y_top


def get_lowest(
    X: Tuple[str],
    y: torch.Tensor,
    scores: torch.Tensor,
    pct: float
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    returns the subset of pct% tweets with smallest scores
    """
    threshold = np.quantile(scores.numpy(), pct)
    bottom_idx = (scores < threshold).tolist()
    X_bottom = tuple(pd.Series(X)[bottom_idx])
    y_bottom = y[bottom_idx]
    return X_bottom, y_bottom


def duplicate_subset(
    X_full: Tuple[str],
    y_full: torch.Tensor,
    X_subset: Tuple[str],
    y_subset: torch.Tensor,
    mixin_method: str
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    augment the training data by duplicating a subset

    :param X_full: tweets
    :param y_full: labels
    :param X_subset: subset of tweets
    :param y_subset: labels of subset of tweets
    :param mixin_method: how to distribute duplicate tweets
    """
    if mixin_method == 'at_end':
        X_with_duplicates = X_full + X_subset
        y_with_duplicates = torch.cat((y_full, y_subset))
    elif mixin_method == 'random':
        X_with_duplicates = tuple(interleave_uniformly(X_full, X_subset))
        y_with_duplicates = torch.Tensor(interleave_uniformly(y_full.tolist(), y_subset.tolist()))
    else:
        raise NotImplementedError(
            f"Parameter 'mixin_method' must be one of ['at_end', 'random'], but got {mixin_method}."
        )
    return X_with_duplicates, y_with_duplicates


def build_curriculum(
    X: Tuple[str],
    y: torch.Tensor,
    hardness_scores: np.array,
    keep_class_distribution: bool,
    shuffle_factor: float
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    return training data sorted by increasing difficulty

    :param X: tweets
    :param y: labels
    :param hardness_scores: measures of classification hardness
    :param keep_class_distribution: whether or not to reorder such that class distributions are preserved
    :param shuffle_factor: how much to disturb curriculum order towards randomness
    """
    bundle = zip(X, y.tolist(), hardness_scores)
    X, y, _ = list(zip(*sorted(bundle, key=lambda tup: tup[2])))
    X, y = tuple(X), torch.Tensor(y)
    if keep_class_distribution:
        X_pos, y_pos = list(zip(*filter(lambda t: t[1] == 1, zip(X, y))))
        X_neg, y_neg = list(zip(*filter(lambda t: t[1] == -1, zip(X, y))))
        X = tuple(interleave_uniformly(X_pos, X_neg, False))
        y = torch.Tensor(interleave_uniformly(list(y_pos), list(y_neg), False))
    if shuffle_factor > 0:
        data = list(zip(list(X), y.tolist()))
        shuffled_data = shuffle_slightly(data, shuffle_factor)
        X, y = zip(*shuffled_data)
        X, y = tuple(X), torch.Tensor(y)
    return X, y


