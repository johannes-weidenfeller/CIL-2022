import pandas as pd
import torch
import random
import numpy as np

from typing import Tuple, Dict


def shuffle(
    X: Tuple[str],
    y: torch.Tensor,
    seed: int
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    shuffle examples based on seed
    
    :param X: tweets
    :param y: labels
    :param seed: seed for RNG
    """
    data = list(zip(X, y.tolist()))
    random.Random(seed).shuffle(data)
    X, y = zip(*data)
    y = torch.Tensor(y)
    return X, y


def ensemble_via_label_mode(
    probabilities: pd.DataFrame
) -> np.array:
    """
    predict mode of predicted labels

    :param probabilities: a dataframe of predicted probabilities for positive labels
    """
    predicted_labels = probabilities.round()
    ensembled_probs = predicted_labels.mean(axis=1).round().values
    return ensembled_probs


def ensemble_via_arithmetic_mean_probability(
    probabilities: pd.DataFrame
) -> np.array:
    """
    predict labels corresponding to arithmetic mean of probabilities

    :param probabilities: a dataframe of predicted probabilities for positive labels
    """
    ensembled_probs = probabilities.mean(axis=1).values
    return ensembled_probs


def ensemble_via_geometric_mean_odds(
    probabilities: pd.DataFrame
) -> np.array:
    """
    predic labels corresponding to softmaxed geometric mean of odds
    
    :param probabilities: a dataframe of predicted probabilities for positive labels
    """
    n = probabilities.shape[1]
    odds = probabilities.divide(1 - probabilities)
    mean_odds = odds.product(axis=1) ** (1 / n)
    ensembled_probs = (mean_odds / (1 + mean_odds)).values
    return ensembled_probs


def ensemble_via_maximum_confidence(
    probabilities: pd.DataFrame
) -> np.array:
    """
    predict labels for which most confidence confidence

    :param probabilities: a dataframe of predicted probabilities for positive labels
    """
    max_conf_getter = lambda x: x[(x - 0.5).abs().idxmax()]
    ensembled_probs = probabilities.apply(max_conf_getter, axis=1).values
    return ensembled_probs


def ensemble_probs(
    probs: pd.DataFrame,
    inference_style: str
) -> np.array:
    """
    makes probabilities from an ensemble of probabilities

    :param probs: a dict of predicted probabilities
    :param inference_style: function identifier
    """
    inference_style_mapper = {
        'pred_mode': ensemble_via_label_mode,
        'prob_mean_arith': ensemble_via_arithmetic_mean_probability,
        'odds_mean_geom': ensemble_via_geometric_mean_odds,
        'conf_max': ensemble_via_maximum_confidence
    }
    return inference_style_mapper[inference_style](probs)


class Ensemble:
    """
    ensembling functionality
    """
    def __init__(
        self,
        model_configs,
        vary_data_order: bool,
        vary_weight_init: bool,
        inference_style: str,
        n_models=None,
    ) -> None:
        """
        initialization

        :param model_configs: either a config dict or list of config dicts
        :param vary_data_order: whether or not to vary data order during training
        :param vary_weight_init: whether or not to vary RNG for weight initialization
        :param inference_style: identifier for inference function
        :param n_models: ignored if model_configs is iterable, otherwise singular config is replicated
        """
        if isinstance(model_configs, dict):
            model_class = model_configs['model_class']
            model_args = model_configs['model_args']
            self.model_classes = [model_class for i in range(n_models)]
            self.model_args = [model_args.copy() for i in range(n_models)]
        else:
            self.model_classes = [c['model_class'] for c in model_configs]
            self.model_args = [c['model_args'].copy() for c in model_configs]
        self.n_models = len(self.model_classes)
        self.vary_data_order = vary_data_order
        self.vary_weight_init = vary_weight_init
        self.inference_style = inference_style

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        """
        train ensemble of models

        :param X: tweets
        :param y: labels
        """
        random.seed(42)
        seeds = random.sample(range(69), self.n_models)
        self.clfs = {}
        for i, seed in enumerate(seeds):
            if self.vary_data_order:
                X, y = shuffle(X, y, seed)
            if self.vary_weight_init:
                self.model_args[i] = {**self.model_args[i], 'manual_seed': seed}
            clf = self.model_classes[i](self.model_args[i])
            clf.fit(X, y)
            self.clfs[i] = clf

    def predict_proba(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        probs = {}
        for i, clf in self.clfs.items():
            probs[i] = clf.predict_proba(X).numpy()[:, 1]
        probs = pd.DataFrame(probs)
        probs = ensemble_probs(probs, self.inference_style)
        probs = np.stack((1 - probs, probs), axis=1)
        return torch.Tensor(probs)

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        return 2 * self.predict_proba(X).round() - 1
