import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel

from typing import Dict, Any, Tuple, List


class BertweetClassifier():
    """ Finetuning BERTweet-base via simpletransformers """
    def __init__(
        self,
        model_args: Dict[str, Any]
    ) -> None:
        self.model = ClassificationModel(
            model_type='bertweet',
            model_name='vinai/bertweet-base',
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
                        self.n_tuple_counters[sentiment][n_tuple] = 0
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
