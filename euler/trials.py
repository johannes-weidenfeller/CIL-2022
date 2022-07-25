import shutil
import random
import gc
import os
import pandas as pd
import torch
import json
from copy import deepcopy

from typing import Any, Callable, Dict, Tuple


from helpers import get_data, evaluate
from preprocessing import vinai_preprocessing, drop_duplicates, reconstruct_smileys, replace_tags
from curriculum import (
    get_mispredictions,
    get_correct_predictions,
    duplicate_subset,
    get_errors,
    get_highest,
    get_lowest,
    build_curriculum
)


def run_trials(
    trials: Dict[str, Callable],
    model_class: Any,
    default_config: Dict[str, Any],
    seed: int,
    train_size: int,
    test_size: int,
    proxy_train_size: int,
    eval_on_train: bool=False,
    out_path: str=None
) -> pd.DataFrame:
    """
    runs a series of experiments on the same training and testing data
    """
    if out_path is not None:
        if not os.path.exists(f'{out_path}/probs'):
            os.makedirs(f'{out_path}/probs')

    # save config
    full_config = {
        'data_order_seed': seed,
        'train_data_size': train_size,
        'test_data_size': test_size,
        'proxy_train_size': proxy_train_size,
        **default_config
    }
    with open(f'{out_path}/full_config.json', 'w') as f:
        json.dump(full_config, f)

    X_train, y_train, X_test, y_test = get_data(seed, train_size + proxy_train_size, test_size)
    X_train, X_train_proxy = X_train[:train_size], X_train[train_size:]
    y_train, y_train_proxy = y_train[:train_size], y_train[train_size:]
    X_train_proxy, y_train_proxy = drop_duplicates(X_train_proxy, y_train_proxy)
    X_train, y_train = drop_duplicates(X_train, y_train)
    X_test, y_test = drop_duplicates(X_test, y_test)

    X_train_proxy = vinai_preprocessing(X_train_proxy)
    X_train = vinai_preprocessing(X_train)
    X_test = vinai_preprocessing(X_test)
    
    proxy_model = model_class(default_config)
    proxy_model.fit(X_train_proxy, y_train_proxy)
    probs = proxy_model.predict_proba(X_train)
    errors = get_errors(y_train, probs)

    torch.save(y_test, f'{out_path}/y_test.pt')
    with open (f'{out_path}/X_test.txt', 'w') as f:
        f.write('\n'.join(X_test))
    trial_names = list(trials.keys())
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    columns = [f'test_{mn}' for mn in metrics]
    if eval_on_train:
        columns = [f'train_{mn}' for mn in metrics] + columns
    results = pd.DataFrame(index=trial_names, columns=columns)
    for trial_name, trial_callable in trials.items():
        trained_model = trial_callable(
            model_class, deepcopy(default_config), deepcopy(X_train), y_train.clone(), errors
        )
        probs_out_path = None if out_path is None else f'{out_path}/probs/{trial_name}.pt'
        res = evaluate(
            trained_model, X_train, y_train, X_test, y_test, eval_on_train, probs_out_path
        )
        for metric_name in metrics:
            for traintest in res.index:
                results.loc[trial_name, f'{traintest}_{metric_name}'] = res.loc[traintest, metric_name]
        results.sort_values(by='test_acc', inplace=True, ascending=False)
        if out_path is not None:
            results.to_csv(f'{out_path}/results.csv')
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree('outputs')
    return results


def with_baseline(
    model_class: Any,
    default_config: Dict[str, Any],
    X: Tuple[str],
    y: torch.Tensor,
    errors: torch.Tensor
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    default configuration

    :param model_class: model class
    :param default_config: default model args
    :param X: tweets
    :param y: labels
    """
    clf = model_class(default_config)
    clf.fit(X, y)
    return clf


def with_subset_duplicated_or_separate(
    model_class: Any,
    default_config: Dict[str, Any],
    X: Tuple[str],
    y: torch.Tensor,
    errors: torch.Tensor,
    duplicate_or_separate: bool,
    subset_type: Callable,
    mixin_method: str=None,
    pct: float=None
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    model trained on some subset (duplicated)

    :param model_class: model class
    :param default_config: default model args
    :param X: tweets
    :param y: labels
    :param duplicate_or_separate: whether to add subset or use subset exclusively
    :param subset_type: identifier for function that creates subset
    :param mixin_method: how to combine full data and subset
    :param pct: percentage of data to use in subset, for non-predictions subset types
    """
    if subset_type == 'lowest_errors':
        X_subset, y_subset = get_lowest(X, y, errors, pct)
    elif subset_type == 'highest_errors':
        X_subset, y_subset = get_highest(X, y, errors, pct)
    else:
        raise NotImplementedError(
            '"subset_type" must be one of ["lowest_errors", "highest_errors"], '
            f'bug got {subset_type} instead.'
        )
    
    if duplicate_or_separate == 'duplicate':
        X, y = duplicate_subset(
            X, y, X_subset, y_subset, mixin_method
        )
    elif duplicate_or_separate == 'separate':
        X, y = X_subset, y_subset
    else:
        raise NotImplementedError(
            '"duplicate_or_separate must be on of ["duplicate", "separate"], '
            f'but got {duplicate_or_separate} instead.'
        )
    clf = model_class(default_config)
    clf.fit(X, y)
    return clf


def with_curriculum_learning(
    model_class: Any,
    default_config: Dict[str, Any],
    X: Tuple[str],
    y: torch.Tensor,
    errors: torch.Tensor,
    keep_class_distribution: bool,
    shuffle_factor: float
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    model trained on a curriculum by increasing difficulty

    :param model_class: model class
    :param default_config: default model args
    :param X: tweets
    :param y: labels
    :param keep_class_distribution: whether or not to reorder
    :param shuffle_facotr: by how much to reintroduce randomness
    """
    X, y = build_curriculum(
        X, y, errors, keep_class_distribution, shuffle_factor
    )
    clf = model_class(default_config)
    clf.fit(X, y)
    return clf


def get_subset_trials(
) -> Dict[str, Callable]:
    """
    get various subset configurations

    :return: dictionary of subset trials
    """
    configs = {
        'easiest-interleaved-10pct': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'lowest_errors', 'mixin_method': 'random', 'pct': 0.1
        },
        'easiest-interleaved-25pct': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'lowest_errors', 'mixin_method': 'random', 'pct': 0.25
        },
        'easiest-interleaved-50pct': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'lowest_errors', 'mixin_method': 'random', 'pct': 0.5
        },
        'hardest-interleaved-10pct': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'highest_errors', 'mixin_method': 'random', 'pct': 0.1
        },
        'hardest-interleaved-25pct': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'highest_errors', 'mixin_method': 'random', 'pct': 0.25
        },
         'hardest-interleaved-50pct': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'highest_errors', 'mixin_method': 'random', 'pct': 0.5
        },
        'no-hard-95pct': {
            'duplicate_or_separate': 'separate', 'subset_type': 'lowest_errors', 'mixin_method': None, 'pct': 0.95
        },
        'no-hard-90pct': {
            'duplicate_or_separate': 'separate', 'subset_type': 'lowest_errors', 'mixin_method': None, 'pct': 0.9
        },
        'no-hard-80pct': {
            'duplicate_or_separate': 'separate', 'subset_type': 'lowest_errors', 'mixin_method': None, 'pct': 0.8
        },
        'no-easy-95pct': {
            'duplicate_or_separate': 'separate', 'subset_type': 'highest_errors', 'mixin_method': None, 'pct': 0.95
        },
        'no-easy-90pct': {
            'duplicate_or_separate': 'separate', 'subset_type': 'highest_errors', 'mixin_method': None, 'pct': 0.95
        },
        'no-easy-80pct': {
            'duplicate_or_separate': 'separate', 'subset_type': 'highest_errors', 'mixin_method': None, 'pct': 0.8
        }
    }
    subset_trials = {}
    for name, config in configs.items():
        def fn(
            model_class, default_config, X, y, errors,
            duplicate_or_separate=config['duplicate_or_separate'],
            subset_type=config['subset_type'],
            mixin_method=config['mixin_method'],
            pct=config['pct']
        ):
            return with_subset_duplicated_or_separate(
                model_class, default_config, X, y, errors,
                duplicate_or_separate, subset_type, mixin_method, pct
            )
        subset_trials[name] = fn
    return subset_trials


def get_curriculum_trials(
) -> Dict[str, Callable]:
    """
    get various curriculum learning configurations

    :return: dictionary of curriculum experiments
    """
    curriculum_configs = {
        'curriculum-reordered-shuffled-00pct': {
            'keep_class_distribution': True, 'shuffle_factor': 0.0
        },
        'curriculum-reordered-shuffled-10pct': {
            'keep_class_distribution': True, 'shuffle_factor': 0.1
        },
        'curriculum-reordered-shuffled-20pct': {
            'keep_class_distribution': True, 'shuffle_factor': 0.2
        },
        'curriculum-reordered-shuffled-30pct': {
            'keep_class_distribution': True, 'shuffle_factor': 0.3
        },
        'curriculum-reordered-shuffled-40pct': {
            'keep_class_distribution': True, 'shuffle_factor': 0.4
        }
    }
    curriculum_trials = {}
    for name, config in curriculum_configs.items():
        def fn(
            model_class, default_config, X, y, errors,
            keep_class_distribution=config['keep_class_distribution'],
            shuffle_factor=config['shuffle_factor']
        ):
            return with_curriculum_learning(
                model_class, default_config, X, y, errors,
                keep_class_distribution, shuffle_factor
            )
        curriculum_trials[name] = fn
    return curriculum_trials


TRIALS = {
    'baseline': with_baseline,
    **get_subset_trials(),
    **get_curriculum_trials()
}


def get_variance_trials(
    n: int
) -> Dict:
    """
    """
    random.seed(42)
    seeds = random.sample(range(69), n)
    variance_trials = {}
    for seed in seeds:
        def fn(model_class, default_config, X, y, errors, seed=seed):
            config = {**default_config, 'manual_seed': seed}
            X, y = shuffle(X, y, seed)
            return with_baseline(model_class, default_config, X, y, errors)
        variance_trials[str(seed)] = fn
    return variance_trials


VARIANCE_TRIALS = {
    'baseline': with_baseline,
    **get_variance_trials(len(TRIALS) - 1)
}




