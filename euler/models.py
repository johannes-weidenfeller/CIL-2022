import tqdm
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier

from typing import Dict, Tuple, Any


class BertweetClassifier:
    """
    finetuning BERTweet-base via simpletransformers
    """
    def __init__(
        self,
        model_args: Dict[str, Any]
    ) -> None:
        """
        initialization

        :param model_args: model hyperparameters
        """
        self.model = ClassificationModel(
            model_type='bertweet',
            model_name='vinai/bertweet-base',
            args=model_args,
            num_labels=2,
            use_cuda=True
        )

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        train_df = pd.DataFrame({'text': X, 'labels': (y + 1) / 2})
        self.model.train_model(train_df=train_df, acc=accuracy_score)
    
    def predict_proba(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        return torch.Tensor(softmax(self.model.predict(X)[1], axis=1))

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        return 2 * self.predict_proba(X)[:, 1].round() - 1


class TokenTupleFrequenciesClassifier:
    def __init__(
        self,
        model_args: Dict[str, Any]
    ) -> None:
        """
        initializes classifier

        :param n: the length of the word tuple
        :param p: the power by which the frequency is weighted within the score
        """
        self.n = model_args['n']
        self.p = model_args['p']

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        """
        computes an occurrence counter for each consecutive n-tuple of words
        for each sentiment, then creates a score by scaling the count by a power
        
        :param X: list of tweets
        :param y: tensor of labels
        """
        self.n_tuple_counters = {-1: dict(), 1: dict()}
        for tweet, label in tqdm.tqdm(zip(X, y), total=len(X), desc='Computing Frequencies'):
            words, sentiment = tweet.split(), label.item()
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
        self.frequencies = {-1: dict(), 1: dict()}
        for sentiment, counter in self.n_tuple_counters.items():
            for n_tuple, count in counter.items():
                self.frequencies[sentiment][n_tuple] = count ** self.p

    def predict_proba(
        self,
        X: Tuple[str],
    ) -> torch.Tensor:
        """
        creates a score for each tweet and each sentiment as the sum of scores for each
        consecutive n-tuple of words in the tweet of that sentiment, calculates the
        probability for positive label as the share of positive score to overall score

        :param X: list of tweets as strings
        :return: a tensor of predicted probabilities for positive label
        """
        probabilities = []
        for tweet in tqdm.tqdm(X, total=len(X), desc='Computing Scores'):
            scores = {-1: 0, 1: 0}
            words = tweet.split()
            if len(words) >= self.n:
                words_offset = []
                for i in range(self.n):
                    words_offset.append(words[i:len(words) - self.n + i + 1])
                for n_tuple_tuple in zip(*words_offset):
                    n_tuple = ' '.join(n_tuple_tuple)
                    for sentiment in [-1, 1]:
                        if n_tuple in self.frequencies[sentiment]:
                            scores[sentiment] += self.frequencies[sentiment][n_tuple]
            if scores[-1] == 0 and scores[1] == 0:
                prob = 0.5
            else:
                prob = scores[1] / (scores[-1] + scores[1])
            probabilities.append(prob)
        return torch.Tensor(probabilities)

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predicts labels

        :param tweets: List of tweets as strings
        :return: a tensor of predicted probabilities for positive label
        """
        probabilities = self.predict_proba(X)
        predictions = probabilities.round() * 2 - 1
        return predictions


class EmbeddingsBoostingClassifier:
    def __init__(
        self,
        model_args: Dict
    ) -> None:
        """
        initialization
        """
        self.val_size = model_args['val_size']
        self.verbose = model_args['verbose']
        self.early_stopping_rounds = model_args['early_stopping_rounds']
        self.embedder = SentenceTransformer(model_args['transformer_name'])
        self.model = CatBoostClassifier(
            early_stopping_rounds=model_args['early_stopping_rounds'],
            task_type='GPU',
            devices='0:1'
        )

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        """
        train cat boost classifier
        """
        X_emb = self.embedder.encode(X)
        y_ = y.numpy()
        split_idx = int(len(X) * (1 - self.val_size))
        X_train, y_train = X_emb[:split_idx], y_[:split_idx]
        X_val, y_val = X_emb[split_idx:], y_[split_idx:]
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=self.verbose
        )

    def predict_proba(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict probabilities
        """
        X_emb = self.embedder.encode(X)
        probs = self.model.predict_proba(X=X_emb)
        return torch.Tensor(probs)

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict labels
        """
        return 2 * self.predict_proba(X)[:, 1].round() - 1
