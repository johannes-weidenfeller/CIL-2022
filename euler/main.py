import os
import warnings
from transformers import logging

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
logging.set_verbosity_error()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from helpers import get_data, evaluate, predict_holdout
from models import BertweetClassifier, TokenTupleFrequenciesClassifier, EmbeddingsBoostingClassifier
from experiments import run_experiments, EXPERIMENTS
from eda import run_eda


SEED = 42
EXPERIMENTAL_TRAIN_SIZE = 25000
DEFAULT_TEST_SIZE = 10000

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


def main(
    run_exploratory_data_analysis=False,
    run_experiments=False,
    run_for_submission=False
) -> None:
    if run_exploratory_data_analysis:
        run_eda()
    if run_experiments:
        experiments_results = run_experiments(
            experiments=EXPERIMENTS,
            model_class=DEFAULT_CLASSIFIER,
            default_config=DEFAULT_MODEL_ARGS,
            seed=SEED,
            train_size=EXPERIMENTAL_TRAIN_SIZE,
            test_size=DEFAULT_TEST_SIZE,
            out_path='experiments_results.csv'
        )

    if run_for_submission:
        pass


if __name__ == '__main__':
    main()
