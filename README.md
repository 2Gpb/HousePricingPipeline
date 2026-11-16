# House Prices

A project for the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Place `train.csv` and `test.csv` in the `raw_data/` directory at the project root.
To run the pipeline from the project root, execute `uv sync --locked`, then `uv run --locked python src/main.py`.
EDA, error analysis and experiment comparison are in `notebooks/draft.ipynb`.

## What was done

This is the second version of my pipeline after Titanic. I wanted to go through a small full development cycle on a regression task and check every step with experiments:

1. EDA: target in log, outliers, duplicate and constant features, missing values plan, ordinal encoding for quality scales.
2. Pipeline: rule-based cleaning is done once for train and test, and everything that uses statistics (missing values, OHE, scaling) is fitted only on the train fold, so there is no leakage. Validation is 5-fold CV with out-of-fold predictions.
3. All classic models with standard parameters: linear models, KNN, trees, random forest and three boostings.
4. MLP in PyTorch with its own training loop and a sklearn-style wrapper, so CV works with it the same way as with Lasso.
5. Error analysis on Lasso OOF, and after it a few experiments with features, one change at a time, compared fold by fold.
6. Ensembles from the saved OOF predictions: mean, blending and stacking.

Every run saves the config snapshot, OOF and test predictions, and adds a row with git hash and CV score to `experiments/log.csv`. A run with a used experiment name fails before training, and `debug: true` skips saving.

## Project structure

Everything starts from `src/main.py`. It reads `src/configs/config.yml` and runs one experiment: loads the data, creates the model by name, runs CV and saves the results. To run another experiment I only change the config.

```
src/
  main.py                  # entry point: data -> model -> CV -> submission, artifacts, log
  configs/
    config.yml             # model, its params, preprocessing flags, experiment name
    config.py              # config loading and paths
  data/
    dataset.py             # cleaning without statistics + preprocessing fitted on the train fold
    predictions.py         # OOF and test predictions of other experiments as features for ensembles
  models/
    factory.py             # creates any model by name from the config
    architectures.py       # PyTorch networks
    estimators.py          # sklearn-style wrapper for a network
    ensembles.py           # mean and blending
  train/
    validation.py          # KFold CV, the same for every model and ensemble
    trainer.py             # training loop for networks
  utils/
    components.py          # optimizer and loss by name
    helpers.py             # tensors, data loader, loss plot
    log_experiment.py      # artifacts, experiment log, git info
    metrics.py, submission.py
```

Dependencies go one way: `main` -> `data`, `models`, `train`, `utils`. The data part doesn't know about models, all differences between models are flags in the config. `run_cv` doesn't know which model it trains, it only calls `fit` and `predict`, so classic models, the MLP and ensembles go through the same code. Torch, LightGBM and XGBoost are imported only when their model is used, because torch and LightGBM/XGBoost crash together in one process on macOS.

Ensembles take the saved OOF and test predictions of other experiments, the base models are not trained again. This works here, because in this competition you only upload a file with predictions. For a competition where you submit code and inference runs on a hidden test, the fold models need to be saved, and there should be a separate inference notebook that loads them, predicts the test and applies the ensemble weights.

## Results

RMSE on log(SalePrice), 5-fold CV, seed=42. Outliers 524 and 1299 are removed in all experiments.
The final model is `mean_lasso_cat_mlp`, chosen by CV.

| Experiment | Model and setup | CV RMSE ± SD | Public RMSE |
| --- | --- | ---: | ---: |
| baseline_v1 | LinearRegression, numeric features only | 0.12879 ± 0.00876 | 0.14456 |
| linreg_full_prep | LinearRegression; full preprocessing: cleaning, ordinal scales, OHE, log1p(LotArea), scaling | 0.12871 ± 0.01113 | 0.14345 |
| ridge_full_prep | Ridge, alpha=1.0 | 0.11782 ± 0.00698 | 0.13697 |
| lasso_full_prep | Lasso, alpha=0.0005 | 0.11116 ± 0.00589 | 0.13288 |
| elastic_net_full_prep | ElasticNet, alpha=0.0005, l1_ratio=0.5 | 0.11245 ± 0.00561 | 0.13325 |
| knn_full_prep | KNN, n_neighbors=10, weights=distance | 0.17092 ± 0.00994 | 0.17524 |
| decision_tree_full_prep | DecisionTree, max_depth=6, min_samples_leaf=5 | 0.19527 ± 0.01017 | 0.16992 |
| random_forest_full_prep | RandomForest, 500 trees, max_features=0.33, min_samples_leaf=2 | 0.13075 ± 0.00722 | 0.14141 |
| cat_boost_full_prep | CatBoost, 2000 iterations, learning_rate=0.03, depth=6, OHE | 0.11330 ± 0.00667 | 0.12477 |
| lightgbm_full_prep | LightGBM, 2000 trees, learning_rate=0.01, num_leaves=15 | 0.11720 ± 0.00771 | 0.12535 |
| xgboost_full_prep | XGBoost, 2000 trees, learning_rate=0.02, max_depth=3 | 0.11323 ± 0.00674 | 0.12570 |
| mlp_full_prep | MLP: 2 layers of 64, GELU, dropout=0.3; AdamW, lr=0.001, weight_decay=0.05, 20 epochs, batch_size=64; standardized target | 0.11611 ± 0.00776 | 0.13051 |
| lasso_log_areas | Lasso + log1p(GrLivArea, OpenPorchSF) | 0.11125 ± 0.00559 | 0.12793 |
| mlp_log_areas | MLP + log1p(GrLivArea, OpenPorchSF) | 0.11570 ± 0.00761 | 0.12777 |
| lasso_ohe_min_freq | Lasso + categories with less than 10 houses merged into one | 0.11175 ± 0.00605 | 0.13180 |
| mlp_ohe_min_freq | MLP + categories with less than 10 houses merged into one | 0.11452 ± 0.00835 | 0.12953 |
| cat_boost_new_features | CatBoost + TotalSF, TotalBath, HouseAge | 0.11259 ± 0.00696 | 0.12351 |
| xgboost_new_features | XGBoost + TotalSF, TotalBath, HouseAge | 0.11306 ± 0.00578 | 0.12554 |
| mean_lasso_cat_mlp | Mean of lasso_full_prep, cat_boost_new_features, mlp_ohe_min_freq | 0.10790 ± 0.00627 | 0.12528 |
| blend_lasso_cat_mlp | Weighted mean of the same models, weights >= 0 with sum 1 fitted on OOF | 0.10830 ± 0.00617 | 0.12530 |
| stack_lasso_cat_mlp | Stacking of the same models, Ridge with positive weights | 0.10815 ± 0.00616 | 0.12524 |

What I took from it:

- Most of the error is in a few cheap houses with abnormal sales (Abnorml, Family). The model is not biased on them, it's just noise, so features can't fix it. Lasso already uses the linear information from numeric features, so new features gave little.
- I compare experiments fold by fold, not only by mean CV. If a change does nothing, the new version is better in each fold with a 50% chance, so 3 of 5 is noise and 5 of 5 is a signal. Only merging rare categories for the MLP was better in 5 of 5.
- Lasso, MLP and boostings make different errors, so the simple mean is better than the best single model in all 5 folds. Blending and stacking are not better than the mean (1 and 2 of 5), so the final model is the mean.
- Public is worse than CV for linear models and ensembles with Lasso, and the best public score is CatBoost alone. I chose the final model by CV, but this gap is worth checking.

## What can be improved

Results and data:

- Find out why linear models are much worse on public than on CV. My guess is extrapolation on very large houses in the test, but I didn't check it.
- Use repeated KFold (several splits with different seeds). With only 5 folds it's hard to tell a small improvement from noise.
- Tune the best models with Optuna, like in Titanic.
- Average several MLPs with different seeds, a network depends a lot on initialization.
- Try Huber loss, because most of the error comes from a few noisy houses.
- Target encoding for Neighborhood inside the fold.

Pipeline:

- Save the fitted fold models and add a separate inference, so the solution works in code competitions.
- `experiments/` is in `.gitignore`, so the log and OOF are only local, and the ensemble can't be built from a clean clone without running the base experiments again. Public scores are also added to the log by hand.
- Log to W&B or MLflow instead of a CSV: fold scores, loss curves, configs and artifacts in one place.
- Use `logging` instead of `print`, so the output has levels and can be saved to a file.
- Change config values from the command line instead of editing `config.yml` for each run.
- Tests for preprocessing, at least that nothing is fitted outside the train fold.
