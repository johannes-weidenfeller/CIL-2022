import pandas as pd
import torch
from emoji import demojize
from nltk.tokenize import TweetTokenizer

from typing import Tuple, Dict, List


def replace_tags(
    tweets: Tuple[str]
) -> Tuple[str]:
    """
    adjust special user and url tokens
    
    :param tweets: tweets
    :return: tweets with tags replaced
    """
    tweets_tokenshandled = []
    for tweet in tweets:
        tokens = []
        for token in tweet.split():
            if token == '<user>':
                tokens.append("@USER")
            elif token == '<url>':
                tokens.append("HTTPURL")
            else:
                tokens.append(token)
        tweets_tokenshandled.append(' '.join(tokens))
    return tuple(tweets_tokenshandled)


def drop_duplicates(
    X: Tuple[str],
    y: torch.Tensor
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    removes duplicate tweets

    :param X: tweets
    :param y: labels
    :return: tuple of unique observations
    """
    X, y = pd.Series(X), y.numpy()
    unique_idx = ~X.duplicated()
    X, y = tuple(X[unique_idx].values.tolist()), torch.Tensor(y[unique_idx].tolist())
    return X, y


def reconstruct_smileys(
    tweets: Tuple[str],
) -> Tuple[str]:
    """
    tries to reconstruct smileys like ":))" or ":(("
    
    :param tweets: tweets
    :return: tweets with reconstructed smileys
    """
    tweets_ = []
    falsealarm_ids = ['<user>', '<url>', 'live at <url>', 'via <user>', 'rt <user>']
    falsealarm_ids = [f'( {s}' for s in falsealarm_ids]

    for tweet in tweets:
        if tweet.count('(') != tweet.count(')'):
            while ') )' in tweet:
                tweet = tweet.replace(') )', ')')
            while '( (' in tweet:
                tweet = tweet.replace('( (', '(')
            if tweet.count('(') > tweet.count(')'):
                falsealarm = False
                for falsealarm_id in falsealarm_ids:
                    if falsealarm_id in tweet:
                        falsealarm = True
                        break
                if falsealarm:
                    tweets_.append(tweet)
                else:
                    tweets_.append(tweet.replace('(', ':('))
            elif '(' not in tweet and ')' in tweet:
                tweets_.append(tweet.replace(')', ':)'))
            else:
                tweets_.append(tweet)
        else:
            tweets_.append(tweet)
    return tuple(tweets_)


def vinai_preprocessing(tweets):
    """
    tokenize using NLTK, translate emotion icons into text strings, normalize; see
    https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
    
    :param tweets: raw tweets
    :return: processed tweets
    """
    replacings = [
        ("cannot ", "can not "),
        ("n't ", " n't "),
        ("n 't ", " n't "),
        ("ca n't", "can't"),
        ("ai n't", "ain't"),
        ("'m ", " 'm "),
        ("'re ", " 're "),
        ("'s ", " 's "),
        ("'ll ", " 'll "),
        ("'d ", " 'd "),
        ("'ve ", " 've "),
        (" p . m .", "  p.m."),
        (" p . m ", " p.m "),
        (" a . m .", " a.m."),
        (" a . m ", " a.m ")
    ]
    tokenizer = TweetTokenizer()
    normalized_tweets = []
    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normalized_tokens = []
        for token in tokens:
            if token == '<user>':
                normalized_token = "@USER"
            elif token == '<url>':
                normalized_token = "HTTPURL"
            elif len(token) == 1:
                normalized_token = demojize(token)
            else:
                if token == "’":
                    normalized_token = "'"
                elif token == "…":
                    normalized_token = "..."
                else:
                    normalized_token = token
            normalized_tokens.append(normalized_token)
        normalized_tweet = ' '.join(normalized_tokens)
        for replacee, replacer in replacings:
            normalized_tweet = normalized_tweet.replace(replacee, replacer)
        normalized_tweet = ' '.join(normalized_tweet.split())
        normalized_tweets.append(normalized_tweet)
    return tuple(normalized_tweets)
