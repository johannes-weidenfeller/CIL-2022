# Tweet Classification

This code was written with Python 3.8.5.  The dependencies can be found in `requirements.txt`.

To run certain components of the project, simply run `main.py` with the respective parameters set to `True`. There are five components that can be run:
- `run_exploratory_data_analysis`: Runs some exploratory data analysis (EDA), specifically, computes sentiment-wise frequencies of certain tweet properties. Outputs are stored in `frequencies_analysis.txt`. See `eda.py` for details. EDA Can be configured via the property `exploratory_data_analysis` in `config.json`. 
- `run_baselines`: Runs the two baseline models and saves results to `/baseline_results`. See `helpers.fit_and_evaluate_baseline_models()` for details. Can be configured via the property `baselines` in `config.json`.
- `run_sensitivity_analysis`: Runs analysis of model performance in change of numerical hyperparameters. Saves results to `/sensitivity_analysis_results`. See `sensitivity_analysis.py` for details. Can be configured via the property `sensitivity_analysis` in `config.json`.
- `run_ensembling_candidate_search`: Runs the ensembling candidate search by fitting a set of models (as specifiable in `ensembling_candidate_trials` in `trials.py`), saving outputs and then evaluating various ensembles on those candidates and then selecting the best ensemble components. Saves results to `/ensembling_search_results`, and results from the benchmark ensembling to `/variance_trials_results`. See `curriculum.py`, `ensembling.py` and `trials.py` for details. Can be configured via the property `ensembling_candidate_search` in `config.json`.   
- `run_full_ensemble`: Runs the full ensemble, that is, trains the chosen best ensembling component models on the full dataset (thus requires having run the ensemble candidate search before or having respective results present). Saves results to `/full_ensemble`. Can be configured via the property `full_ensemble` in `config.py`. 
