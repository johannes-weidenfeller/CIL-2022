import torch
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import AdamWeightDecay
from transformers import TFAutoModelForSequenceClassification as TFModel

from typing import Dict, Any, Tuple, List


class BertweetClassifier:
    """ Finetuning BERTweet-base via simpletransformers """
    def __init__(
            self,
            model_args: Dict[str, Any],
            model_base='vinai/bertweet-base'
    ) -> None:
        checkpoint = model_args['checkpoint'] if 'checkpoint' in model_args else None
        name = checkpoint or model_base
        self.model = ClassificationModel(
            model_type='bertweet',
            model_name=name,
            tokenizer_name=model_base,
            tokenizer_type=AutoTokenizer.from_pretrained(model_base),
            args=model_args,
            num_labels=2
        )

    def fit(
            self,
            X: Tuple[str],
            y: torch.Tensor
    ) -> None:
        train_df = pd.DataFrame({'text': X, 'labels': (y + 1) / 2})
        self.model.train_model(train_df, acc=accuracy_score)

    def predict(
            self,
            X: Tuple[str]
    ) -> torch.Tensor:
        return 2 * torch.Tensor(self.model.predict(X)[0]) - 1


class BertweetLargeClassifier(BertweetClassifier):
    def __init__(self, model_args):
        super().__init__(model_args, 'vinai/bertweet-large')


class BaselineModel:
    def __init__(
            self,
            model_args: Dict[str, Any]
    ) -> None:
        """
        :param n: the length of the word tuple
        :param p: the power by which the frequency is weighted within the score
        """
        self.n = model_args['n']
        self.p = model_args['p']

    def fit(
            self,
            X: List[str],
            y: torch.Tensor
    ) -> None:
        """
        Computes an occurrence counter for each consecutive n-tuple of words
        for each sentiment. Then creates a score by scaling the count by a power.

        :param X: list of tweets
        :param y: tensor of labels
        """
        self.n_tuple_counters = {-1: dict(), 1: dict()}
        for tweet, label in zip(X, y):
            words, sentiment = tweet.split(' '), label.item()
            if len(words) >= self.n:
                words_offset = []
                for i in range(self.n):
                    words_offset.append(words[i:len(words) - self.n + i + 1])
                for n_tuple_tuple in zip(*words_offset):
                    n_tuple = ' '.join(n_tuple_tuple)
                    if n_tuple in self.n_tuple_counters[sentiment]:
                        self.n_tuple_counters[sentiment][n_tuple] += 1
                    else:
                        self.n_tuple_counters[sentiment][n_tuple] = 1
        self.scores = {-1: dict(), 1: dict()}
        for sentiment, counter in self.n_tuple_counters.items():
            for n_tuple, count in counter.items():
                self.scores[sentiment][n_tuple] = count ** self.p

    def predict_proba(
            self,
            X: List[str],
    ) -> torch.Tensor:
        """
        Creates a score for each tweet and each sentiment as the sum of scores for each
        consecutive n-tuple of words in the tweet of that sentiment. Calculates the
        probability for positive label as the share of positive score to overall score.

        :param X: List of tweets as strings
        :return: a tensor of predicted probabilities for positive label
        """
        pos_probabilities = []
        for tweet in X:
            neg_score, pos_score = 0, 0
            words = tweet.split(' ')
            if len(words) >= self.n:
                words_offset = []
                for i in range(self.n):
                    words_offset.append(words[i:len(words) - self.n + i + 1])
                for n_tuple_tuple in zip(*words_offset):
                    n_tuple = ' '.join(n_tuple_tuple)
                    if n_tuple in self.scores[-1]:
                        neg_score += self.scores[-1][n_tuple]
                    if n_tuple in self.scores[1]:
                        pos_score += self.scores[1][n_tuple]
            scores_sum = neg_score + pos_score
            if scores_sum == 0:
                pos_prob = 0.5
            else:
                pos_prob = pos_score / scores_sum
            pos_probabilities.append(pos_prob)
        return torch.Tensor(pos_probabilities)

    def predict(
            self,
            X: List[str]
    ) -> torch.Tensor:
        """
        predicts labels

        :param tweets: List of tweets as strings
        :return: a tensor of predicted probabilities for positive label
        """
        probabilities = self.predict_proba(X)
        predictions = probabilities.round() * 2 - 1
        return predictions


class Ensemble:
    """
    very simple ensembling over mode of predicted labels, using varing train data order
    """
    def __init__(self, model_class, model_args, n_models):
        assert n_models % 2 == 1
        self.model_class = model_class
        self.model_args = model_args
        self.n_models = n_models

    def fit(
            self,
            X: Tuple[str],
            y: torch.Tensor
    ) -> None:
        seeds = random.sample(range(69), self.n_models)
        self.clfs = {}
        for seed in seeds:
            data = list(zip(X, y.tolist()))
            random.Random(seed).shuffle(data)
            X_, y_ = zip(*data)
            y_ = torch.Tensor(y_)
            clf = self.model_class(self.model_args)
            clf.fit(X_, y_)
            self.clfs[seed] = clf

    def predict(
            self,
            X: Tuple[str]
    ) -> torch.Tensor:
        preds = {}
        for seed, clf in self.clfs.items():
            preds[seed] = clf.predict(X)
        preds = pd.DataFrame(preds)
        preds = torch.Tensor(preds.mode(axis=1).loc[:, 0])
        return preds


class LearningRateWarmupCooldown(tf.keras.callbacks.Callback):
    def __init__(
            self,
            model: TFModel,
            peak_lr: float,
            warmup_ratio: float,
            n_training_steps: int,
            initial_lr: float=0,
            final_lr: float=0
    ) -> None:
        super(LearningRateWarmupCooldown, self).__init__()
        self.model = model
        self.peak_lr = peak_lr
        self.warmup_ratio = warmup_ratio
        self.n_training_steps = n_training_steps
        self.peak_step = warmup_ratio * n_training_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_train_batch_end(self, batch, logs=None):
        if batch < self.peak_step:
            delta = batch / self.peak_step
            lr = delta * self.peak_lr + (1 - delta) * self.initial_lr
        else:
            delta = (batch - self.peak_step) / (self.n_training_steps - self.peak_step)
            lr = delta * self.final_lr + (1 - delta) * self.peak_lr
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


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
        self.weight_decay = model_args['weight_decay']
        self.dropout = model_args['dropout']
        self.warmup_ratio = model_args['warmup_ratio']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def fit(
            self,
            X: Tuple[str],
            y: torch.Tensor
    ) -> None:
        inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices((dict(inputs), (y + 1) / 2))
        ds = dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.model = TFModel.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout
        )
        self.model.compile(
            optimizer=AdamWeightDecay(
                learning_rate=self.learning_rate,
                weight_decay_rate=self.weight_decay
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        n_training_steps = len(X) // self.batch_size * self.epochs + 1
        lr_scheduler = LearningRateWarmupCooldown(
            model=self.model,
            peak_lr=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            n_training_steps=n_training_steps
        )
        self.history = self.model.fit(ds, epochs=self.epochs, callbacks=[lr_scheduler])

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