import shutil
import gc
import os
import pandas as pd
import torch
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
from ensembling import Ensemble


def run_experiments(
    experiments: Dict[str, Callable],
    model_class: Any,
    default_config: Dict[str, Any],
    seed: int,
    train_size: int,
    test_size: int,
    eval_on_train: bool=False,
    out_path: str=None
) -> pd.DataFrame:
    """
    runs a series of experiments on the same training and testing data
    """
    if out_path is not None:
        if not os.path.exists(f'{out_path}/probs'):
            os.makedirs(f'{out_path}/probs')
    X_train, y_train, X_test, y_test = get_data(seed, train_size, test_size)
    torch.save(y_test, f'{out_path}/y_test.pt')
    with open (f'{out_path}/X_test.txt', 'w') as f:
        f.write('\n'.join(X_test))
    experiment_names = list(experiments.keys())
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    columns = [f'test_{mn}' for mn in metrics]
    if eval_on_train:
        columns = [f'train_{mn}' for mn in metrics] + columns
    results = pd.DataFrame(index=experiment_names, columns=columns)
    for experiment_name, experiment_callable in experiments.items():
        trained_model, X_train_, y_train_, X_test_ = experiment_callable(
            model_class, deepcopy(default_config), deepcopy(X_train), y_train.clone(), deepcopy(X_test)
        )
        probs_out_path = None if out_path is None else f'{out_path}/probs/{experiment_name}.pt'
        res = evaluate(trained_model, X_train_, y_train_, X_test_, y_test, eval_on_train, probs_out_path)
        for metric_name in metrics:
            for traintest in res.index:
                results.loc[experiment_name, f'{traintest}_{metric_name}'] = res.loc[traintest, metric_name]
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
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str]
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    default configuration

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    """
    clf = model_class(default_config)
    clf.fit(X_train, y_train)
    return clf, X_train, y_train, X_test


def with_vinai_preprocessing(
    model_class: Any,
    default_config: Dict[str, Any],
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str]
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    standard preprocessing as suggested by VinAIResearch for BERTweet

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    """
    X_train_preprocessed = vinai_preprocessing(X_train)
    X_test_preprocessed = vinai_preprocessing(X_test)
    clf = model_class(default_config)
    clf.fit(X_train_preprocessed, y_train)
    return clf, X_train_preprocessed, y_train, X_test_preprocessed


def with_only_unique_tweets(
    model_class: Any,
    default_config: Dict[str, Any],
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str]
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    drop dulicate tweets

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    """
    X_train_unique, y_train_unique = drop_duplicates(X_train, y_train)
    clf = model_class(default_config)
    clf.fit(X_train, y_train)
    return clf, X_train_unique, y_train_unique, X_test


def with_reconstructed_smileys(
    model_class: Any,
    default_config: Dict[str, Any],
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str]
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    reconstruct smileys

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    """
    X_train_with_smileys = reconstruct_smileys(X_train)
    X_test_with_smileys = reconstruct_smileys(X_test)
    clf = model_class(default_config)
    clf.fit(X_train_with_smileys, y_train)
    return clf, X_train_with_smileys, y_train, X_test_with_smileys


def with_replaced_tags(
    model_class: Any,
    default_config: Dict[str, Any],
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str]
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    replace tags

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    """
    X_train_with_tags = replace_tags(X_train)
    X_test_with_tags = replace_tags(X_test)
    clf = model_class(default_config)
    clf.fit(X_train_with_tags, y_train)
    return clf, X_train_with_tags, y_train, X_test_with_tags



def with_subset_duplicated_or_separate(
    model_class: Any,
    default_config: Dict[str, Any],
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str],
    duplicate_or_separate: bool,
    subset_type: Callable,
    mixin_method: str=None,
    pct: float=None
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    model trained on some subset (duplicated)

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    :param duplicate_or_separate: whether to add subset or use subset exclusively
    :param subset_type: identifier for function that creates subset
    :param mixin_method: how to combine full data and subset
    :param pct: percentage of data to use in subset, for non-predictions subset types
    """
    clf = model_class(default_config)
    clf.fit(X_train, y_train)
    if subset_type in {'mispredictions', 'correct_predictions'}:
        preds = clf.predict(X_train)
    elif subset_type in {'lowest_errors', 'highest_errors'}:
        probs = clf.predict_proba(X_train)
        errors = get_errors(y_train, probs)
    else:
        raise NotImplementedError(
            '"subset_type" must be one of '
            '["mispredictions", "correct_predictions", "lowest_errors", "highest_errors"], '
            f'but got {subset_type} instead.'
        )

    if subset_type == 'mispredictions':
        X_subset, y_subset = get_mispredictions(X_train, y_train, preds)
    elif subset_type == 'correct_predictions':
        X_subset, y_subset = get_correct_predictions(X_train, y_train, preds)
    elif subset_type == 'lowest_errors':
        X_subset, y_subset = get_lowest(X_train, y_train, errors, pct)
    elif subset_type == 'highest_errors':
        X_subset, y_subset = get_highest(X_train, y_train, errors, pct)

    if duplicate_or_separate == 'duplicate':
        X_train_, y_train_ = duplicate_subset(
            X_train, y_train, X_subset, y_subset, mixin_method
        )
    elif duplicate_or_separate == 'separate':
        X_train_, y_train_ = X_subset, y_subset
    else:
        raise NotImplementedError(
            '"duplicate_or_separate must be on of ["duplicate", "separate"], '
            f'but got {duplicate_or_separate} instead.'
        )
    print(len(X_train_), subset_type, duplicate_or_separate, mixin_method)
    clf = model_class(default_config)
    clf.fit(X_train_, y_train_)
    return clf, X_train, y_train, X_test


def with_curriculum_learning(
    model_class: Any,
    default_config: Dict[str, Any],
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str],
    keep_class_distribution: bool,
    shuffle_factor: float
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    model trained on a curriculum by increasing difficulty

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    :param keep_class_distribution: whether or not to reorder
    :param shuffle_facotr: by how much to reintroduce randomness
    """
    clf = model_class(default_config)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_train)
    errors = get_errors(y_train, probs)
    X_train_curr, y_train_curr = build_curriculum(X_train, y_train, errors, keep_class_distribution, shuffle_factor)
    clf = model_class(default_config)
    clf.fit(X_train_curr, y_train_curr)
    return clf, X_train, y_train, X_test


def with_ensemble_learning(
    model_class: Any,
    default_config: Dict[str, Any],
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str],
    vary_data_order: bool,
    vary_weight_init: bool,
    inference_style: str,
    n_models: int=None
) -> Tuple[Any, Tuple[str], torch.Tensor, Tuple[str]]:
    """
    model trained on a curriculum by increasing difficulty

    :param model_class: model class
    :param default_config: default model args
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    :param vary_data_order: whether to vary train data order
    :param vary_weight_init: whether to vary seed for weight init RNG
    :param inference_style: how to ensemble probabilities
    :param n_models: how many models
    """
    model_config = {'model_class': model_class, 'model_args': default_config}
    clf = Ensemble(model_config, vary_data_order, vary_weight_init, inference_style, n_models)
    clf.fit(X_train, y_train)
    return clf, X_train, y_train, X_test


def get_subset_experiments(
) -> Dict[str, Callable]:
    """
    get various subset configurations

    :return: dictionary of subset experiments
    """
    configs = {
        'mispredictions-appended': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'mispredictions', 'mixin_method': 'at_end', 'pct': None
        },
        'mispredictions-interleaved': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'mispredictions', 'mixin_method': 'random', 'pct': None
        },
        'corrects-appended': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'correct_predictions', 'mixin_method': 'at_end', 'pct': None
        },
        'corrects-interleaved': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'correct_predictions', 'mixin_method': 'random', 'pct': None
        },
        'easiest-appended': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'lowest_errors', 'mixin_method': 'at_end', 'pct': 0.1
        },
        'easiest-interleaved': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'lowest_errors', 'mixin_method': 'random', 'pct': 0.1
        },
        'hardest-appended': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'highest_errors', 'mixin_method': 'at_end', 'pct': 0.1
        },
        'hardest-interleaved': {
            'duplicate_or_separate': 'duplicate', 'subset_type': 'highest_errors', 'mixin_method': 'random', 'pct': 0.1
        },
        'only-mispredictions': {
            'duplicate_or_separate': 'separate', 'subset_type': 'mispredictions', 'mixin_method': None, 'pct': None
        },
        'only-corrects': {
            'duplicate_or_separate': 'separate', 'subset_type': 'correct_predictions', 'mixin_method': None, 'pct': None
        },
        'no-hard': {
            'duplicate_or_separate': 'separate', 'subset_type': 'lowest_errors', 'mixin_method': None, 'pct': 0.9
        },
        'no-easy': {
            'duplicate_or_separate': 'separate', 'subset_type': 'highest_errors', 'mixin_method': None, 'pct': 0.9
        }
    }
    subset_experiments = {}
    for name, config in configs.items():
        def fn(
            mc, dc, Xtr, yt, Xte,
            duplicate_or_separate=config['duplicate_or_separate'],
            subset_type=config['subset_type'],
            mixin_method=config['mixin_method'],
            pct=config['pct']
        ):
            return with_subset_duplicated_or_separate(
                mc, dc, Xtr, yt, Xte,
                duplicate_or_separate, subset_type, mixin_method, pct
            )
        subset_experiments[name] = fn
    return subset_experiments


def get_preprocessing_experiments(
) -> Dict[str, Callable]:
    """
    get various preprocessing configurations

    :return: dictionary of preprocessing experiments
    """
    preprocessing_experiments =  {
        'baseline': with_baseline,
        'vinai-preprocessing': with_vinai_preprocessing,
        'drop-duplicates': with_only_unique_tweets,
        'smiley-reconstruction': with_reconstructed_smileys,
        'tags-replacement': with_replaced_tags
    }
    return preprocessing_experiments


def get_curriculum_experiments(
) -> Dict[str, Callable]:
    """
    get various curriculum learning configurations

    :return: dictionary of curriculum experiments
    """
    curriculum_configs = {
        'curriculum-strict': {
            'keep_class_distribution': False, 'shuffle_factor': 0
        },
        'curriculum-reordered': {
            'keep_class_distribution': True, 'shuffle_factor': 0
        },
        'curriculum-strict-shuffled': {
            'keep_class_distribution': False, 'shuffle_factor': 0.2
        },
        'curriculum-reordered-shuffled': {
            'keep_class_distribution': True, 'shuffle_factor': 0.2
        }
    }
    curriculum_experiments = {}
    for name, config in curriculum_configs.items():
        def fn(
            mc, dc, Xtr, yt, Xte,
            keep_class_distribution=config['keep_class_distribution'],
            shuffle_factor=config['shuffle_factor']
        ):
            return with_curriculum_learning(
                mc, dc, Xtr, yt, Xte, keep_class_distribution, shuffle_factor
            )
        curriculum_experiments[name] = fn
    return curriculum_experiments


def get_ensembling_experiments(
) -> Dict[str, Callable]:
    """
    get various ensemble learning configurations

    :return: dictionary of ensembling experiments
    """
    ensembling_configs = {
        'dataorder-mode-3': {
            'vary_data_order': True, 'vary_weight_init': False, 'inference_style': 'pred_mode', 'n_models': 3
        },
        'weightinit-mode-3': {
            'vary_data_order': False, 'vary_weight_init': True, 'inference_style': 'pred_mode', 'n_models': 3
        },
        'dataorder-weightinit-mode-3': {
            'vary_data_order': True, 'vary_weight_init': True, 'inference_style': 'pred_mode', 'n_models': 3
        },
        'dataorder-weightinit-arith-3': {
            'vary_data_order': True, 'vary_weight_init': True, 'inference_style': 'prob_mean_arith', 'n_models': 3
        },
        'dataorder-weightinit-geom-3': {
            'vary_data_order': True, 'vary_weight_init': True, 'inference_style': 'odds_mean_geom', 'n_models': 3
        },
        'dataorder-weightinit-conf-3': {
            'vary_data_order': True, 'vary_weight_init': True, 'inference_style': 'conf_max', 'n_models': 3
        }
    }
    ensembling_experiments = {}
    for name, config in ensembling_configs.items():
        def fn(
            mc, dc, Xtr, yt, Xte,
            vary_data_order=config['vary_data_order'],
            vary_weight_init=config['vary_weight_init'],
            inference_style=config['inference_style'],
            n_models=config['n_models']
        ):
            return with_ensemble_learning(
                mc, dc, Xtr, yt, Xte,
                vary_data_order, vary_weight_init, inference_style, n_models
            )
        ensembling_experiments[name] = fn
    return ensembling_experiments


def get_hyperparam_experiments(
) -> Dict[str, Callable]:
    """
    get various hyperparameter configurations

    :return: dictionary of hyperparameter experiments
    """
    hyperparam_configs = {
        'smaller-batches-smaller-learning-rate': {
            'train_batch_size': 12, 'learning_rate': 1e-5
        },
        'smaller-batches-larger-learning-rate': {
            'train_batch_size': 12, 'learning_rate': 4e-5
        },
        'larger-batches-smaller-learning-rate': {
            'train_batch_size': 24, 'learning_rate': 1e-5
        },
        'larger-batches-larger-learning-rate': {
            'train_batch_size': 24, 'learning_rate': 4e-5
        }
    }
    hyperparam_experiments = {}
    for name, config in hyperparam_configs.items():
        def fn(mc, dc, Xtr, yt, Xte, hyperparam_overwrites=config):
            updated_config = {**dc, **hyperparam_overwrites}
            return with_baseline(mc, updated_config, Xtr, yt, Xte)
        hyperparam_experiments[name] = fn
    return hyperparam_experiments


EXPERIMENTS = {
    **get_preprocessing_experiments(),
    **get_subset_experiments(),
    **get_curriculum_experiments(),
    #**get_ensembling_experiments(),
    **get_hyperparam_experiments(),
}
