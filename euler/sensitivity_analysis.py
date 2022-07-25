import pandas as pd
import json
import os
import torch
import shutil
import gc

from helpers import get_data, evaluate
from preprocessing import drop_duplicates, vinai_preprocessing

from typing import Any, Dict, List


def run_sensitivity_analysis(
    model_class: Any,
    default_config: Dict[str, Any],
    seed: int,
    train_size: int,
    test_size: int,
    numerical_model_args: List[str],
    factor: float=2.0,
    out_path: str=None
) -> pd.DataFrame:
    """
    analyize change of hyperparameter on performance
    """
    if out_path is not None:
        if not os.path.exists(f'{out_path}/probs'):
            os.makedirs(f'{out_path}/probs')
    
    full_config = {
        'data_order_seed': seed,
        'train_data_size': train_size,
        'test_data_size': test_size,
        'numerical_model_args': numerical_model_args,
        'factor': factor,
        **default_config
    }
    with open(f'{out_path}/full_config.json', 'w') as f:
        json.dump(full_config, f)

    X_train, y_train, X_test, y_test = get_data(seed, train_size, test_size)
    X_train, y_train = drop_duplicates(X_train, y_train)
    X_test, y_test = drop_duplicates(X_test, y_test)
    X_train, X_test = vinai_preprocessing(X_train), vinai_preprocessing(X_test)

    configs = {'default': {**default_config}}
    for param in numerical_model_args:
        value = default_config[param]
        neighbors = [(1 / factor) * value, factor * value]
        neighbors = [type(value)(val) for val in neighbors]
        for val in neighbors:
            configs[f'{param}-{val:.2g}'] = {**default_config, param: val}

    torch.save(y_test, f'{out_path}/y_test.pt')
    with open (f'{out_path}/X_test.txt', 'w') as f:
        f.write('\n'.join(X_test))
    names = list(configs.keys())
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    columns = [f'test_{mn}' for mn in metrics]
    results = pd.DataFrame(index=names, columns=columns)
    for name, config in configs.items():
        trained_model = model_class(config)
        trained_model.fit(X_train, y_train)
        probs_out_path = None if out_path is None else f'{out_path}/probs/{name}.pt'
        res = evaluate(
            trained_model, X_train, y_train, X_test, y_test, False, probs_out_path
        )
        for metric_name in metrics:
            results.loc[name, f'test_{metric_name}'] = res.loc['test', metric_name]
        results.sort_values(by='test_acc', inplace=True, ascending=False)
        if out_path is not None:
            results.to_csv(f'{out_path}/results.csv')
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree('outputs')
    return results

