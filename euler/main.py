import os
import json
import gc
import torch
import random
import pandas as pd
import warnings
from copy import deepcopy
from transformers import logging

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
logging.set_verbosity_error()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Dict, Any, Tuple, List

from helpers import get_data, evaluate, predict_holdout, run_best, get_holdout
from models import BertweetClassifier, TokenTupleFrequenciesClassifier, EmbeddingsBoostingClassifier
from preprocessing import vinai_preprocessing, drop_duplicates
from curriculum import get_errors
from trials import run_trials, TRIALS, VARIANCE_TRIALS
from eda import run_eda
from ensembling import ensembling_candidate_search, choose_ensembling_components, get_ensemble_components, shuffle
from sensitivity_analysis import run_sensitivity_analysis


SEED = 42
DEFAULT_TEST_SIZE = 10000
PROXY_TRAIN_SIZE = 10000
TRIALS_OUT_PATH = 'trials_results'
EXPERIMENTS_OUT_PATH = 'experiment_results'
EVAL_ON_TRAIN = False
DEFAULT_CLASSIFIER = BertweetClassifier
DEFAULT_MODEL_ARGS = {
    'manual_seed': 69,
    'num_train_epochs': 1,
    'train_batch_size': 16,
    'learning_rate': 2e-5,
    'dropout': 0.1,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'scheduler': 'linear_schedule_with_warmup',
    'optimizer': 'AdamW',
    'adam_epsilon': 1e-6,
    'max_grad_norm': 1.0,
    'overwrite_output_dir': True,
    'save_steps': -1,
    'save_model_every_epoch': False
}

EMBEDDINGS_BOOSTING_CLASSIFIER_CONFIG = {
    'val_size': 0.1,
    'early_stopping_rounds': 16,
    'transformer_name': 'r2d2/stsb-bertweet-base-v0',
    'verbose': False
}

TOKEN_TUPLE_FREQUENCIES_CLASSIFIER_CONFIG = {
    'n': 2,
    'p': 0.25
}

ENSEMBLING_SEARCH_TRAIN_SIZE = 180000
ENSEMBLING_SEARCH_OUT_PATH = 'ensembling_search_results'
ENSEMBLING_SEARCH_N_MODELS_LIST = [2, 3, 4, 5]
ENSEMBLING_SEARCH_INFERENCE_STYLES = ['pred_mode', 'prob_mean_arith', 'odds_mean_geom']
ENSEMBLING_SEARCH_QUANTILE = 0.16

SENSITIVITY_ANALYSIS_TRAIN_SIZE = 190000
SENSITIVITY_ANALYSIS_NUMERICAL_MODEL_ARGS = ['train_batch_size', 'learning_rate', 'dropout', 'weight_decay', 'warmup_ratio']
SENSITIVITY_ANALYSIS_FACTOR = 2.0
SENSITIVITY_ANALYSIS_OUT_PATH = 'sensitivity_analysis_results'

FULL_TRAIN_SIZE = 2480000


def main(
    run_exploratory_data_analysis=False,
    run_ensembling_trials_suite=False,
    run_ensembling_candidate_search=False,
    run_for_submission=False,
    run_hyperparameters_sensitivity_analysis=False
) -> None:
    if run_exploratory_data_analysis:
        run_eda()
    if run_ensembling_trials_suite:
        ensembling_trials_results = run_trials(
            trials=TRIALS,
            model_class=DEFAULT_CLASSIFIER,
            default_config=DEFAULT_MODEL_ARGS,
            seed=SEED,
            train_size=ENSEMBLING_SEARCH_TRAIN_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            proxy_train_size=PROXY_TRAIN_SIZE,
            eval_on_train=EVAL_ON_TRAIN,
            out_path=ENSEMBLING_SEARCH_OUT_PATH
        )
    if run_ensembling_candidate_search:
        ensembling_candidate_search(
            out_path=ENSEMBLING_SEARCH_OUT_PATH,
            n_models_list=ENSEMBLING_SEARCH_N_MODELS_LIST,
            inference_styles=ENSEMBLING_SEARCH_INFERENCE_STYLES
        )
        choose_ensembling_components(
            n_models_list=ENSEMBLING_SEARCH_N_MODELS_LIST,
            inference_styles=ENSEMBLING_SEARCH_INFERENCE_STYLES,
            out_path=ENSEMBLING_SEARCH_OUT_PATH,
            pct=ENSEMBLING_SEARCH_QUANTILE
        )
    if run_hyperparameters_sensitivity_analysis:
        run_sensitivity_analysis(
            model_class=DEFAULT_CLASSIFIER,
            default_config=DEFAULT_MODEL_ARGS,
            seed=SEED,
            train_size=SENSITIVITY_ANALYSIS_TRAIN_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            numerical_model_args=SENSITIVITY_ANALYSIS_NUMERICAL_MODEL_ARGS,
            factor=SENSITIVITY_ANALYSIS_FACTOR,
            out_path=SENSITIVITY_ANALYSIS_OUT_PATH
        )
    if run_for_submission:
        run_best(
            seed=SEED,
            train_size=FULL_TRAIN_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            model_class=DEFAULT_CLASSIFIER,
            model_args=DEFAULT_MODEL_ARGS,
            out_path='example_submission'
        )

def fit_ensemble_models(
    out_path: str,
    unique_tweets_only: bool,
    preprocess: bool,
    save_models
):
    if os.path.exists(out_path):
        raise AssertionError(f'"{out_path}" already exists, choose another path!')
    os.makedirs(out_path)

    # get data
    X_train, y_train, X_test, y_test = get_data(SEED, FULL_TRAIN_SIZE + PROXY_TRAIN_SIZE, DEFAULT_TEST_SIZE)
    X_train, X_train_proxy = X_train[:FULL_TRAIN_SIZE], X_train[FULL_TRAIN_SIZE:]
    y_train, y_train_proxy = y_train[:FULL_TRAIN_SIZE], y_train[FULL_TRAIN_SIZE:]
    X = get_holdout()

    # (preprocess)
    if unique_tweets_only:
        X_train_proxy, y_train_proxy = drop_duplicates(X_train_proxy, y_train_proxy)
        X_train, y_train = drop_duplicates(X_train, y_train)
        X_test, y_test = drop_duplicates(X_test, y_test)
    
    if preprocess:
        X_train_proxy = vinai_preprocessing(X_train_proxy)
        X_train = vinai_preprocessing(X_train)
        X_test = vinai_preprocessing(X_test)
        X = vinai_preprocessing(X)
    
    # train proxy model
    print(f'training proxy model for estimating difficulties:')
    proxy_model = DEFAULT_CLASSIFIER(DEFAULT_MODEL_ARGS)
    proxy_model.fit(X_train_proxy, y_train_proxy)
    probs = proxy_model.predict_proba(X_train)
    print('estimating difficulties:')
    errors = get_errors(y_train, probs)

    # get ensemble components
    ensemble_components = get_ensemble_components(
        component_candidates=TRIALS,
        ensemble_search_path=ENSEMBLING_SEARCH_OUT_PATH
    )

    random.seed(SEED)
    seeds = random.sample(range(69), len(ensemble_components))
    for i, (model_name, model_callable) in enumerate(ensemble_components.items()):
        print(f'fitting model {i + 1} out of {len(ensemble_components)} ({model_name}):')
        os.makedirs(f'{out_path}/{model_name}')

        # use different seed for each model
        seed = seeds[i]
        X_train_shuffled, y_train_shuffled = shuffle(deepcopy(X_train), y_train.clone(), seed)
        config = {**DEFAULT_MODEL_ARGS, 'manual_seed': seed}
        if save_models:
            config['output_dir'] = f'{out_path}/{model_name}/outputs'
        clf = model_callable(
            model_class=DEFAULT_CLASSIFIER,
            default_config=config,
            X=X_train_shuffled,
            y=y_train_shuffled,
            errors=errors
        )
        # save submission
        probabilities = clf.predict_proba(X)
        predictions = 2 * probabilities[:, 1].round() - 1
        predictions = predictions.to(int).tolist()
        ids = range(1, len(X) + 1)
        submission = pd.DataFrame([ids, predictions], index=['Id', 'Prediction']).T
        submission.to_csv(f'{out_path}/{model_name}/submission.csv', index=False)
        torch.save(probabilities, f'{out_path}/{model_name}/probabilities.pt')

        # evaluate on test set
        res = evaluate(
            clf, X_train, y_train, X_test, y_test, False, f'{out_path}/{model_name}/test_probabilities.pt'
        )
        res.to_csv(f'{out_path}/{model_name}/test_results.csv')
        
        # save config
        component_config = {
            'data_order_seed': seed,
            'component_type': model_name,
            **config
        }
        with open(f'{out_path}/{model_name}/config.json', 'w') as f:
            json.dump(component_config, f)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    #main()
    fit_ensemble_models(
        out_path='full_ensemble',
        unique_tweets_only=True,
        preprocess=False,
        save_models=True
    )





