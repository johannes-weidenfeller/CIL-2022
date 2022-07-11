import argparse

from helpers import pipeline
from models import BertweetClassifier, BertweetLargeClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--num-epochs', '-ep', type=int, default=1)
    parser.add_argument('--batch-size', '-bs', type=int, default=16)
    parser.add_argument('--test-size', '-ts', type=float, default=0.3)
    parser.add_argument('--learning-rate', '-lr', type=float, default=2e-5)
    parser.add_argument('--checkpoint', '-cp', type=str, default=None)
    return parser.parse_args()


def main(args):
    model_class = BertweetLargeClassifier if args.large else BertweetClassifier
    model_args = {
        'manual_seed': 69,
        'num_train_epochs': args.num_epochs,
        'train_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': 0.1,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'optimizer': 'AdamW',
        'overwrite_output_dir': True,
        'checkpoint': args.checkpoint
    }

    pipeline(
        full=args.full,
        seed=42,
        test_size=args.test_size,
        model_class=model_class,
        model_args=model_args,
        out_path='submission.csv'
    )


if __name__ == '__main__':
    main(parse_args())
