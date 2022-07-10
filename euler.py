import urllib.request
import zipfile
import io
import os
import random

import torch
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification as TFModel

from typing import Tuple, Dict, Any


def download_data():
    urllib.request.urlretrieve(DATA_URL, ZIP_FNAME)


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
    zf = zipfile.ZipFile('twitter-datasets.zip')
    prefix = 'twitter-datasets/train_'
    suffix = '_full' if full else ''

    with io.TextIOWrapper(
            zf.open(f'{prefix}neg{suffix}.txt'), encoding='utf-8'
    ) as f:
        neg = f.read().split('\n')[:-1]
    with io.TextIOWrapper(
            zf.open(f'{prefix}pos{suffix}.txt'), encoding='utf-8'
    ) as f:
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


class TFBertweetClassifier():
    """ finetuning pretrained models via tensorflow """
    def __init__(
            self,
            model_args: Dict[str, Any]
    ) -> None:
        self.model_name = model_args['model_name']
        self.epochs = model_args['epochs']
        self.batch_size = model_args['batch_size']
        self.learning_rate = model_args['learning_rate']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            X: Tuple[str],
            y: torch.Tensor
    ) -> None:
        inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), (y + 1) / 2))
        ds = dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.model = TFModel.from_pretrained(self.model_name, num_labels=2)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.history = self.model.fit(ds, epochs=self.epochs)

    def predict_proba(
            self,
            X: Tuple[str]
    ) -> tf.Tensor:
        inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices((dict(inputs),)).batch(self.batch_size)
        probs = tf.nn.softmax(self.model.predict(dataset)['logits'])
        return probs

    def predict(
            self,
            X: Tuple[str]
    ) -> torch.Tensor:
        probs = self.predict_proba(X)
        preds = tf.argmax(probs, axis=1) * 2 - 1
        return torch.Tensor(preds.numpy())


def main():
    DATA_URL = 'http://www.da.inf.ethz.ch/files/twitter-datasets.zip'
    ZIP_FNAME = 'twitter-datasets.zip'

    if not os.path.exists(ZIP_FNAME):
        urllib.request.urlretrieve(DATA_URL, ZIP_FNAME)

    FULL = False
    SEED = 42
    TEST_SIZE = 0.3

    ts = 100
    model_class = TFBertweetClassifier
    model_args = {
        'model_name': 'vinai/bertweet-base',
        'epochs': 1,
        'batch_size': 16,
        'val_size': 0.1,
        'learning_rate': 2e-5,
    }

    X_train, y_train, X_test, y_test = get_data(FULL, SEED, TEST_SIZE)
    X_train, y_train, X_test, y_test = X_train[:ts], y_train[:ts], X_test[:ts], y_test[:ts]

    clf = model_class(model_args)
    clf.fit(X_train, y_train)

    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)

    train_acc = calc_clf_acc(y_train, y_train_hat)
    test_acc = calc_clf_acc(y_test, y_test_hat)

    print(f'Training accuracy: {train_acc * 100:.2f}%')
    print(f'Testing accuracy: {test_acc * 100:.2f}%')


if __name__ == '__main__':
    gpu_slot = 0
    with tf.device('/GPU:' + str(gpu_slot)):
        main()
