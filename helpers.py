import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from emoji import demojize
from nltk.tokenize import TweetTokenizer

from typing import Tuple, Dict, Any, List, Union

from models import BertweetClassifier


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


def sensitivity_analysis(
    model_class: Any,
    config: Dict,
    param_name: str,
    param_values: List[Union[int, float]],
    n: int
) -> None:
    """
    inspects the changes of accuracy in varying parameter value

    :param model_clss: model class
    :param config: model arguments
    :param param_name: name of parameter to investivate
    :param param_values: list of parameter values
    :param n: (default) size of training and testing data
    """
    df = pd.DataFrame(index=param_values, columns=['train', 'test'])
    X_train, y_train, X_test, y_test = get_data(False, 42, 0.3)
    X_test, y_test = X_test[:n], y_test[:n]
    for param_value in param_values:
        params = config if param_name == 'train_size' else {**config, param_name: param_value}
        clf = model_class(params)
        ts = param_value if param_name == 'train_size' else n
        clf.fit(X_train[:ts], y_train[:ts])
        df.loc[param_value, 'train'] = calc_clf_acc(y_train[:ts], clf.predict(X_train[:ts]))
        df.loc[param_value, 'test'] = calc_clf_acc(y_test, clf.predict(X_test))
    df.to_csv(f'sensitivity_analysis/{param_name}_sensitivity.csv')


def plot_sensitivity_analysis_results(
    sensitivity_params: Dict[str, List[Union[float, int]]]
) -> None:
    """
    plots sensitivities of accuracies to parameter value changes on log-plot.
    Assumes an odd number of values per parameter, default value in the middle.

    :param sensitivity_params: a dict of param_name, param_values pairs
    """
    n = len(sensitivity_params)
    fig, ax = plt.subplots(1, n, figsize=(8 * n, 4))
    for i, param_name in enumerate(sensitivity_params.keys()):
        df = pd.read_csv(f'sensitivity_analysis/{param_name}_sensitivity.csv', index_col=0)
        ax[i].set_xscale('log')
        ax[i].set_xticks(df.index)
        ax[i].set_xticklabels(df.index)
        for ds in df.columns:
            ax[i].plot(df.index, df[ds], label=ds)
        default = df.index.values[df.shape[0] // 2]
        ax[i].axvline(default, color='red', label='default')
        ax[i].set_title(f'Accuracy in {param_name}')
        ax[i].set_xlabel(param_name)
        ax[i].set_ylabel('Accuracy')
        ax[i].legend()
    plt.show()


def normalizeToken(token):
    if token == '<user>':
        return "@USER"
    elif token == '<url>':
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tokenizer, tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


def normalizeTweets(tweets):
    """ see https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py """
    tokenizer = TweetTokenizer()
    return tuple(normalizeTweet(tokenizer, t) for t in tweets)