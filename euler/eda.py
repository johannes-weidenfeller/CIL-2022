import torch
import pandas as pd

from typing import Tuple, Dict, Any, List, Union, Callable

from helpers import get_data
from preprocessing import drop_duplicates


def frequencies_eda(
    tweets: Tuple[str],
    labels: torch.Tensor,
    get_property: Callable,
    level: str
) -> pd.DataFrame:
    """
    computes frequencies of tweet or token properties by sentiment

    :param tweets: tweets
    :param labels: labels
    :param get_property: callable returning property or None
    :param level: either 'tweet' or 'token'
    :return: dataframe with property index, columns pos, neg, total, pos_prob
    """
    frequencies = {-1: {}, 1: {}}
    for tweet, label in zip(tweets, labels.tolist()):
        if level == 'tweet':
            prop = get_property(tweet)
            if prop is not None:
                if prop in frequencies[label]:
                    frequencies[label][prop] += 1
                else:
                    frequencies[label][prop] = 1
        elif level == 'token':
            for token in tweet.split():
                prop = get_property(token)
                if prop is not None:
                    if prop in frequencies[label]:
                        frequencies[label][prop] += 1
                    else:
                        frequencies[label][prop] = 1
        else:
            raise NotImplementedError(
                f'"level" must be "tweet" or "token", but got {level}'
            )
    idx = list(set(frequencies[-1].keys()) | set(frequencies[1].keys()))
    props = pd.DataFrame(0, index=idx, columns=[-1, 1])
    for sentiment, freq in frequencies.items():
        for prop, count in freq.items():
            props.loc[prop, sentiment] += count
    props['total'] = props.sum(axis=1)
    props['pos_prob'] = props[1] / props['total']
    props.sort_values(by='total', ascending=False, inplace=True)
    props.rename(columns={-1: 'neg', 1: 'pos'}, inplace=True)
    return props


def length_prop(tweet):
    """ tweet length rounded to 10 """
    bracket = len(tweet) // 10
    return f'{bracket * 10} - {bracket * 10 + 9}'


def hashtag_prop(token):
    """ hashtag tokens """
    if token.startswith('#'):
        return token
    return None


def tags_prop(token):
    """ tag tokens """
    if token == '<user>' or token == '<url>':
        return token
    return None


def onechartoken_prop(token):
    """ single-character tokens """
    if len(token) == 1:
        return token
    return None


def run_eda(n=10):
    """
    run some frequency analysis
    """
    tweets, labels, _, _ = get_data(42, 2490000, 10000)
    
    tweets_freq = frequencies_eda(tweets, labels, lambda t: t, 'tweet')
    print(f'\nTop {n} Most Frequent Tweets:')
    print(tweets_freq[:n])

    tweets, labels = drop_duplicates(tweets, labels)

    lengths_freq = frequencies_eda(tweets, labels, length_prop, 'tweet')
    print(f'\nTop {n} Tweet Lengths:')
    print(lengths_freq[:n])
    
    tokens_freq = frequencies_eda(tweets, labels, lambda t: t, 'token')
    tokens_freq_common = tokens_freq[tokens_freq['total'] > len(tweets) * 0.01]
    print(f'\nTop {n} Most Negative Common Tokens:')
    print(tokens_freq_common.sort_values(by='pos_prob')[:n])
    print(f'\nTop {n} Most Positive Common Tokens:')
    print(tokens_freq_common.sort_values(by='pos_prob', ascending=False)[:n])

    hashtags_freq = frequencies_eda(tweets, labels, hashtag_prop, 'token')
    print(f'\nTop {n} Most Frequent Hashtags:')
    print(hashtags_freq[:n])

    tags_freq = frequencies_eda(tweets, labels, tags_prop, 'token')
    print('\nSpecial Tokens:')
    print(tags_freq)

    onechartokens_freq = frequencies_eda(tweets, labels, onechartoken_prop, 'token')
    print(f'\nTop {n} Single-Character Tokens:')
    print(onechartokens_freq[:n])

