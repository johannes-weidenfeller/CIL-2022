import urllib.request
import zipfile
import io
import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef, roc_curve, auc, average_precision_score
)

from typing import Tuple, Dict, Any, List, Union


def download_data():
    """
    downlaods data if not stored on disk already
    """
    data_url = 'http://www.da.inf.ethz.ch/files/twitter-datasets.zip'
    data_fname = 'twitter-datasets.zip'
    if not os.path.exists(data_fname):
        urllib.request.urlretrieve(data_url, data_fname)


def get_data(
    seed: int,
    train_size: int,
    test_size: int
) -> Tuple[Tuple[str], torch.Tensor, Tuple[str], torch.Tensor]:
    """
    loads training and testing data

    :param seed: a seed to determine the shuffling
    :param train_size: the absolute size of the train set
    :param test_size: the absolute size of the test set
    :return: a tuple (X_train, y_train, X_test, y_test)
    """
    download_data()
    suffix = '_full' if train_size + test_size > 2e5 else ''
    class_size = (train_size + test_size) // 2

    neg_path = f'twitter-datasets/train_neg{suffix}.txt'
    pos_path = f'twitter-datasets/train_pos{suffix}.txt'
    
    with zipfile.ZipFile('twitter-datasets.zip') as zf:
        with io.TextIOWrapper(zf.open(neg_path), encoding='utf-8') as f:
            neg = f.read().split('\n')[:-1][:class_size]
        with io.TextIOWrapper(zf.open(pos_path), encoding='utf-8') as f:
            pos = f.read().split('\n')[:-1][:class_size]
    
    data = list(zip(neg, [-1] * len(neg))) + list(zip(pos, [1] * len(pos)))
    random.Random(seed).shuffle(data)

    train, test = data[:train_size], data[train_size:]

    X_train, y_train = list(zip(*train))
    X_test, y_test = list(zip(*test))
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    return X_train, y_train, X_test, y_test


def Xy_to_df(
    X: Tuple[str],
    y: torch.Tensor
) -> pd.DataFrame:
    """
    converts obervations from (tuple, tensor) pairs to dataframe

    :param X: tweets
    :param y: labels
    :return: a dataframe with observations in rows
    """
    return pd.DataFrame({'text': X, 'labels': (y + 1) / 2})


def calc_metrics(
    sentiments: torch.Tensor,
    probs: torch.Tensor
) -> Dict[str, float]:
    """
    compute metrics
    (very heavily inspired by simpletransformers.ClassificationModel.compute_metrics)

    :param sentiments: classes
    :param probs: predited probabilities
    """
    labels = sentiments.numpy() / 2 + 0.5
    preds = np.argmax(probs.numpy(), axis=1)
    pos_probs = probs.numpy()[:, 1]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr, tpr, _ = roc_curve(labels, pos_probs)
    metrics = {
        'acc': accuracy_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'auroc': auc(fpr, tpr),
        'auprc': average_precision_score(labels, pos_probs)
    }
    return metrics


def evaluate(
    model: Any,
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str],
    y_test: torch.Tensor,
    out_path: str=None
) -> pd.DataFrame:
    """
    evaluate model on train and test set for various metrics

    :param model: a model
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    :param y_test: testing labels
    :param out_path: if not None, save results to out_path
    """
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    train_res = calc_metrics(y_train, train_probs)
    test_res = calc_metrics(y_test, test_probs)
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    results = pd.DataFrame(index=['train', 'test'], columns=metrics)
    for traintest, res in [('train', train_res), ('test', test_res)]:
        for metric in metrics:
            results.loc[traintest, metric] = res[metric]
    if out_path is not None:
        results.to_csv(out_path)
    return results


def predict_holdout(
    clf: Any,
    out_path: str
) -> None:
    """
    predicts on the holdout "test_data.txt" using provided classifier

    :param clf: classifier with .predict() method
    :param out_path: where to save output to
    """
    path = 'twitter-datasets/test_data.txt'
    with zipfile.ZipFile('twitter-datasets.zip') as zf:
        with io.TextIOWrapper(zf.open(path), encoding='utf-8') as f:
            X = tuple([','.join(l.split(',')[1:]) for l in f.read().split('\n')[:-1]])
    ids = range(1, len(X) + 1)
    predictions = clf.predict(X).to(int).tolist()
    submission = pd.DataFrame([ids, predictions], index=['Id', 'Prediction']).T
    submission.to_csv(out_path, index=False)


def merge_experiment_results(
    fnames: List[str],
    out_path: str
) -> None:
    """
    merge a set of results files into one
    """
    dfs = []
    for fn in fnames:
        df = pd.read_csv(fn, index_col=0)
        experiment_type = fn.split('_')[0]
        df.insert(loc=0, column='experiment-type', value=experiment_type)
        dfs.append(df)
    res = pd.concat(dfs)
    res.sort_values(by='test_acc', ascending=False, inplace=True)
    frontcols = ['experiment-type', 'train_acc', 'test_acc']
    res = res[frontcols + [c for c in res.columns if c not in frontcols]]
    res.to_csv(out_path)

