import tqdm
import pandas as pd
import torch
import random
import numpy as np
from scipy.special import softmax
import itertools

from typing import Tuple, Dict

import helpers


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


def ensembling_candidate_search(
    out_path: str,
    performance_threshold: float,
    n_models: int,
    inference_style: str
) -> pd.DataFrame:
    """
    tries ensembling all combinations of n_models on all models exceeding performance_threshold

    :param out_path: path of experiment results, containing test set and probabilities
    :param performance_threshold: consider only models exceeding this threshold
    :param n_models: number of models per ensemble
    :param inference_style: how to ensemble predictions
    """
    results = pd.read_csv(f'{out_path}/results.csv', index_col=0)
    candidates = results[results.test_acc > performance_threshold].index.tolist()
    with open(f'{out_path}/X_test.txt', 'r') as f:
        X_test = tuple([f.read().split('\n')])
    y_test = torch.load(f'{out_path}/y_test.pt')
    candidates = results[results.test_acc > performance_threshold].index.tolist()
    probs = {
        candidate: torch.load(f'{out_path}/probs/{candidate}.pt') for candidate in candidates
    }
    combinations = list(itertools.combinations(candidates, n_models))
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    candidate_search_results = pd.DataFrame(
        index=range(len(combinations)),
        columns=[f'model {i}' for i in range(1, n_models + 1)] + metrics
    )
    for i, combination in tqdm.tqdm(enumerate(combinations), total=len(combinations)):
        pos_probs = pd.DataFrame(
            {model_id: probs[model_id].numpy()[:, 1] for model_id in combination}
        )
        ensembled_probs = ensemble_probs(pos_probs, inference_style)
        test_probs = torch.Tensor(np.stack((1 - ensembled_probs, ensembled_probs), axis=1))
        metrics_res = helpers.calc_metrics(y_test, test_probs)
        for j in range(n_models):
            candidate_search_results.loc[i, f'model {j + 1}'] = combination[j]
        for metric in metrics:
            candidate_search_results.loc[i, metric] = metrics_res[metric]
    candidate_search_results.sort_values(by='acc', ascending=False, inplace=True)
    candidate_search_results.to_csv(f'{out_path}/candidate_search_results-{inference_style}-{n_models}.csv')
    return candidate_search_results


def rank_ensembling_candidates(
    candidate_search_results: pd.DataFrame,
    pct: float,
    subset_size: int,
    out_path: str
) -> pd.DataFrame:
    """
    ranks subsets of models by frequency in top qantile

    :param candidate_search_result: result by ensembling_candidate_search()
    :param pct: indicating which share of best models to consider
    :param subset_size: number of models to consider together
    :param out_path: where to save results to
    """
    df = candidate_search_results.sort_values(by='acc', ascending=False)
    n = df.shape[0]
    model_cols = [c for c in df.columns if 'model ' in c]
    ensembling_size = len(model_cols)
    top_models = df[:int(n * pct)][model_cols]
    ensembles = set(top_models.to_numpy().flatten())
    ensemble_subsets = list(itertools.combinations(ensembles, subset_size))
    ensemble_subsets = [tuple(sorted(ss)) for ss in ensemble_subsets]
    frequencies = {ensemble_subset: 0 for ensemble_subset in ensemble_subsets}
    for i, models in top_models.iterrows():
        for subset in list(itertools.combinations(models.tolist(), subset_size)):
            frequencies[tuple(sorted(subset))] += 1
    frequencies = sorted(frequencies.items(), key=lambda t: t[1], reverse=True)
    best_subsets = [modelnames for modelnames, _ in frequencies]
    counts = [count for _, count in frequencies]
    results = pd.DataFrame({'count': counts})
    results[[f'model {i}' for i in range(1, subset_size + 1)]] = best_subsets
    results.to_csv(f'{out_path}/best_candidates_{subset_size}_of_{ensembling_size}.csv')
    return results

