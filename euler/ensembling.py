import tqdm
import pandas as pd
import torch
import json
import os
import random
import numpy as np
from scipy.special import softmax
import itertools

from typing import Tuple, Dict, Set, List, Callable

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
    n_models_list: List[int],
    inference_styles: List[str]
):
    """
    tries ensembling all combinations of n models with inference style inference_stlye
    for each n in n_models_list and each inference_style in inference_styles

    :param out_path: path of experiment results, containing test set and probabilities
    :param n_models: number of models per ensemble
    :param inference_style: how to ensemble predictions
    """
    results = pd.read_csv(f'{out_path}/results.csv', index_col=0)
    with open(f'{out_path}/X_test.txt', 'r') as f:
        X_test = tuple([f.read().split('\n')])
    y_test = torch.load(f'{out_path}/y_test.pt')
    candidates = results.index.tolist()
    probs = {
        candidate: torch.load(f'{out_path}/probs/{candidate}.pt') for candidate in candidates
    }
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    if not os.path.exists(f'{out_path}/candidate_search'):
        os.makedirs(f'{out_path}/candidate_search')
    for n_models in n_models_list:
        if not os.path.exists(f'{out_path}/candidate_search/ensemble_{n_models}'):
            os.makedirs(f'{out_path}/candidate_search/ensemble_{n_models}')
        combinations = list(itertools.combinations(candidates, n_models))
        for inference_style in inference_styles:
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
            path = f'{out_path}/candidate_search/ensemble_{n_models}/{inference_style}.csv'
            candidate_search_results.to_csv(path)
    

def choose_ensembling_components(
    n_models_list: List[int],
    inference_styles: List[str],
    out_path: str,
    pct: float,
) -> pd.DataFrame:
    """

    :param pct: indicating which share of best models to consider
    :param out_path: where to save results to
    """
    candidates_search_summary = pd.DataFrame(index=n_models_list, columns=inference_styles)
    best_ensembles_summary = dict()
    for n_models in n_models_list:
        best_ensembles_summary[n_models] = dict()
        for inference_style in tqdm.tqdm(inference_styles, total=len(inference_styles)):
            path = f'{out_path}/candidate_search/ensemble_{n_models}/{inference_style}.csv'
            df = pd.read_csv(path, index_col=0)
            df.sort_values(by='acc', ascending=True)
            k = df.shape[0]
            model_cols = [c for c in df.columns if 'model ' in c]
            top_models = df[:int(k * pct)][model_cols]
            ensembles = set(top_models.to_numpy().flatten())
            ensemble_subsets = list(itertools.combinations(ensembles, n_models - 1))
            ensemble_subsets = [tuple(sorted(subset)) for subset in ensemble_subsets]
            frequencies = {subset: 0 for subset in ensemble_subsets}
            for i, models in top_models.iterrows():
                subset_occurrences = list(itertools.combinations(models.tolist(), n_models - 1))
                for subset in subset_occurrences:
                    frequencies[tuple(sorted(subset))] += 1
            frequencies = sorted(frequencies.items(), key=lambda t: t[1], reverse=True)
            best_components = set(frequencies[0][0])
            last_component_candidates = ensembles - best_components
            last_component_candidate_scores = {candidate: 0 for candidate in last_component_candidates}
            subsets = list(itertools.combinations(best_components, n_models - 2))
            subsets = [set(subset) for subset in subsets]
            for i, models in top_models.iterrows():
                models_set = set(models.tolist())
                for subset in subsets:
                    if subset < models_set:
                        for candidate in models_set - best_components:
                            last_component_candidate_scores[candidate] += 1
            last_component_candidate_scores = sorted(
                last_component_candidate_scores.items(), key=lambda t: t[1], reverse=True
            )
            best_last_candidate = last_component_candidate_scores[0][0]
            best_components.add(best_last_candidate)
            best_components = sorted(list(best_components))
            idx = df[model_cols].apply(lambda c: set(c.tolist()), axis=1) == set(best_components)
            best_ensemble_score = df[idx]['acc'].item()
            best_ensembles_summary[n_models][inference_style] = {
                'components': best_components, 'accuracy': best_ensemble_score
            }
            top_models_score = df[:int(k * pct)]['acc'].mean()
            candidates_search_summary.loc[n_models, inference_style] = top_models_score
    candidates_search_summary.to_csv(f'{out_path}/candidate_search_summary.csv')
    with open(f'{out_path}/best_ensembles_summary.json', 'w') as f:
        json.dump(best_ensembles_summary, f)


def get_ensemble_components(
    component_candidates: Dict[str, Callable],
    ensemble_search_path: str
):
    """
    """
    with open(f'{ensemble_search_path}/best_ensembles_summary.json', 'r') as f:
        best_ensembles_summary = json.load(f)
    n = max(best_ensembles_summary.keys())
    best_acc = 0
    best_components = []
    for inference_method, res in best_ensembles_summary[n].items():
        if res['accuracy'] > best_acc:
            best_acc = res['accuracy']
            best_components = res['components']
    components = {}
    for component in best_components:
        components[component] = component_candidates[component]
    return components

