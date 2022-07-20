import os
import warnings
from transformers import logging

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
logging.set_verbosity_error()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd

from typing import Dict, Any, Tuple, List

from helpers import get_data, evaluate, predict_holdout, run_best
from models import BertweetClassifier, TokenTupleFrequenciesClassifier, EmbeddingsBoostingClassifier
from experiments import run_experiments, EXPERIMENTS
from eda import run_eda
from ensembling import ensembling_candidate_search, rank_ensembling_candidates


SEED = 42
EXPERIMENTAL_TRAIN_SIZE = 25000
DEFAULT_TEST_SIZE = 10000
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


ENSEMBLING_SEARCH_PERFORMANCE_THRESHOLD = 0.9
ENSEMBLING_SEARCH_N_MODELS = 4
ENSEMBLING_SEARCH_INFERENCE_STYLE = 'prob_mean_arith'
ENSEMBLING_SEARCH_QUANTILE = 0.1
ENSEMBLING_SEARCH_SUBSET_SIZE = 3


def main(
    run_exploratory_data_analysis=False,
    run_experiments_suite=False,
    run_ensembling_candidate_search=False,
    run_for_submission=False
) -> None:
    if run_exploratory_data_analysis:
        run_eda()
    if run_experiments_suite:
        experiments_results = run_experiments(
            experiments=EXPERIMENTS,
            model_class=DEFAULT_CLASSIFIER,
            default_config=DEFAULT_MODEL_ARGS,
            seed=SEED,
            train_size=EXPERIMENTAL_TRAIN_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            eval_on_train=EVAL_ON_TRAIN,
            out_path=EXPERIMENTS_OUT_PATH
        )
    if run_ensembling_candidate_search:
        candidate_search_results = ensembling_candidate_search(
            out_path=EXPERIMENTS_OUT_PATH,
            performance_threshold=ENSEMBLING_SEARCH_PERFORMANCE_THRESHOLD,
            n_models=ENSEMBLING_SEARCH_N_MODELS,
            inference_style=ENSEMBLING_SEARCH_INFERENCE_STYLE
        )
        best_candidates = rank_ensembling_candidates(
            candidate_search_results=candidate_search_results,
            pct=ENSEMBLING_SEARCH_QUANTILE,
            subset_size=ENSEMBLING_SEARCH_SUBSET_SIZE,
            out_path=EXPERIMENTS_OUT_PATH
        )
    if run_for_submission:
        run_best(
            seed=SEED,
            train_size=190000,
            test_size=10000,
            model_class=DEFAULT_CLASSIFIER,
            model_args=DEFAULT_MODEL_ARGS,
            out_path='example_submission'
        )


if __name__ == '__main__':
    main(run_for_submission=True)


