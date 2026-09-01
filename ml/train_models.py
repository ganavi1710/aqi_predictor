"""
Multi-Model Training, Cross-Validation, Hyperparameter Optimization, and Benchmark Pipeline.
Trains and compares 7 regression algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Extra Trees Regressor
6. Support Vector Regression (SVR - with StandardScaler Pipeline)
7. K-Nearest Neighbors (KNN - with StandardScaler Pipeline)
"""

import os
import sys
import time
import json
import pickle
import datetime
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score

from ml.preprocessing import clean_dataset, impute_missing_values, prepare_features, split_data
from ml.evaluate_models import (
    compute_regression_metrics,
    generate_model_diagnostic_plots,
    generate_feature_importance_comparison
)
from ml.model_selection import select_best_model, export_model_comparison, print_model_comparison_table

def run_multi_model_pipeline(
    data_path: str = None,
    models_dir: str = None,
    results_dir: str = None,
    static_plots_dir: str = None
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'air_quality_data.csv')
    if models_dir is None:
        models_dir = os.path.join(base_dir, 'models')
    if results_dir is None:
        results_dir = os.path.join(base_dir, 'results')
    if static_plots_dir is None:
        static_plots_dir = os.path.join(base_dir, 'static', 'models')

    eval_plots_dir = os.path.join(models_dir, 'evaluation_plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(eval_plots_dir, exist_ok=True)
    os.makedirs(static_plots_dir, exist_ok=True)

    print("\n" + "="*70)
    print("      SMART AQI - MULTI-MODEL MACHINE LEARNING PIPELINE       ")
    print("="*70)

    # 1. Load Dataset
    print(f"[1/7] Loading dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    print(f"      Initial shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # 2. Preprocess & Clean
    print("\n[2/7] Cleaning dataset & imputing missing values...")
    df_clean = clean_dataset(df_raw)
    df_clean = impute_missing_values(df_clean)

    # 3. Features & Target
    print("\n[3/7] Selecting features & target variable...")
    X, y, feature_names = prepare_features(df_clean)
    print(f"      Features ({len(feature_names)}): {feature_names}")
    print("      Target: AQI (strictly excluded from input features)")

    # 4. Train / Test Split (80% train / 20% test on same seed)
    print("\n[4/7] Splitting dataset (80% train / 20% test, random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 5. Model Configurations & Hyperparameter Grids
    # For SVR and KNN, StandardScaler is embedded in a Pipeline to prevent data leakage.
    algorithms_config = {
        'linear_regression': {
            'name': 'Linear Regression',
            'estimator': LinearRegression(),
            'param_grid': {},
            'is_pipeline': False
        },
        'decision_tree': {
            'name': 'Decision Tree Regressor',
            'estimator': DecisionTreeRegressor(random_state=42),
            'param_grid': {
                'max_depth': [6, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [2, 4],
                'criterion': ['squared_error', 'friedman_mse']
            },
            'is_pipeline': False
        },
        'random_forest': {
            'name': 'Random Forest Regressor',
            'estimator': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [8, 12],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'is_pipeline': False
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting Regressor',
            'estimator': GradientBoostingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 4]
            },
            'is_pipeline': False
        },
        'extra_trees': {
            'name': 'Extra Trees Regressor',
            'estimator': ExtraTreesRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [8, 12],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'is_pipeline': False
        },
        'svr': {
            'name': 'Support Vector Regression (SVR)',
            'estimator': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR())
            ]),
            'param_grid': {
                'svr__C': [10.0, 100.0, 300.0],
                'svr__epsilon': [0.1, 0.5],
                'svr__kernel': ['rbf']
            },
            'is_pipeline': True
        },
        'knn': {
            'name': 'K-Nearest Neighbors Regressor (KNN)',
            'estimator': Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsRegressor())
            ]),
            'param_grid': {
                'knn__n_neighbors': [3, 5, 7, 9],
                'knn__weights': ['uniform', 'distance']
            },
            'is_pipeline': True
        }
    }

    # 6. Model Training, 5-Fold Cross-Validation & Test Evaluation
    print("\n[5/7] Training & Tuning Models with 5-Fold Cross-Validation...")
    models_metrics = {}
    trained_models = {}

    for key, config in algorithms_config.items():
        name = config['name']
        print(f"\n   --> Training {name}...")
        start_time = time.time()

        if config['param_grid']:
            grid = GridSearchCV(
                estimator=config['estimator'],
                param_grid=config['param_grid'],
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            cv_scores = -grid.best_score_  # Convert negative RMSE to positive
            # Calculate standard deviation across splits
            cv_std = float(np.std(grid.cv_results_['split0_test_score'] + grid.cv_results_['split1_test_score']))
        else:
            # Baseline without grid search (Linear Regression)
            model_inst = config['estimator']
            cv_raw = cross_val_score(model_inst, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
            cv_scores = float(-np.mean(cv_raw))
            cv_std = float(np.std(cv_raw))
            model_inst.fit(X_train, y_train)
            best_model = model_inst
            best_params = {"fit_intercept": True}

        train_duration = round(time.time() - start_time, 3)
        trained_models[key] = best_model

        # Evaluate on the SAME unseen test set
        y_test_pred = best_model.predict(X_test)
        test_metrics = compute_regression_metrics(y_test, y_test_pred)

        # Extract feature importances if available
        feat_importances = {}
        inner_estimator = best_model.named_steps[list(best_model.named_steps.keys())[-1]] if hasattr(best_model, 'named_steps') else best_model
        if hasattr(inner_estimator, 'feature_importances_'):
            feat_importances = {
                f: round(float(imp), 4)
                for f, imp in zip(feature_names, inner_estimator.feature_importances_)
            }

        models_metrics[key] = {
            'name': name,
            'best_hyperparameters': best_params,
            'cv_mean_rmse': round(float(cv_scores), 3),
            'cv_std_rmse': round(float(cv_std), 3),
            'training_time_sec': train_duration,
            'test_metrics': test_metrics,
            'feature_importances': feat_importances
        }

        print(f"       [OK] Test RMSE: {test_metrics['RMSE']:.3f} | R2: {test_metrics['R2_Score']:.4f} | Time: {train_duration}s")

    # 7. Automated Best Model Selection
    print("\n[6/7] Automatically determining the best-performing model...")
    best_key, best_info = select_best_model(models_metrics)
    print(f"      [*] BEST MODEL SELECTED: {best_info['name']}")
    print(f"          - Test RMSE: {best_info['test_metrics']['RMSE']}")
    print(f"          - Test MAE:  {best_info['test_metrics']['MAE']}")
    print(f"          - Test R2:   {best_info['test_metrics']['R2_Score']}")
    print(f"          - 5-Fold CV: {best_info['cv_mean_rmse']} (+/- {best_info['cv_std_rmse']})")

    # Export comparison results to CSV and JSON
    df_summary = export_model_comparison(models_metrics, results_dir, best_key)
    print_model_comparison_table(df_summary, best_key)

    # 8. Generate Evaluation & Diagnostic Plots
    print("\n[7/7] Generating diagnostic plots and exporting artifacts...")
    tree_models_for_plot = {}
    for key, model_obj in trained_models.items():
        generate_model_diagnostic_plots(key, algorithms_config[key]['name'], model_obj, X_test, y_test, eval_plots_dir)
        generate_model_diagnostic_plots(key, algorithms_config[key]['name'], model_obj, X_test, y_test, static_plots_dir)

        if key in ['decision_tree', 'random_forest', 'gradient_boosting', 'extra_trees']:
            tree_models_for_plot[algorithms_config[key]['name']] = model_obj

    # Grouped feature importance comparison chart
    generate_feature_importance_comparison(tree_models_for_plot, feature_names, eval_plots_dir)
    generate_feature_importance_comparison(tree_models_for_plot, feature_names, static_plots_dir)

    # Save individual model PKLs
    for key, model_obj in trained_models.items():
        pkl_path = os.path.join(models_dir, f'{key}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_obj, f)

    # Save best model to root model.pkl and models.pkl for backward compatibility
    best_model_obj = trained_models[best_key]
    with open(os.path.join(base_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model_obj, f)
    with open(os.path.join(models_dir, 'decision_tree_aqi.pkl'), 'wb') as f:
        pickle.dump(trained_models.get('decision_tree', best_model_obj), f)

    compat_dict = {k: m for k, m in trained_models.items()}
    compat_dict['best_model'] = best_model_obj
    with open(os.path.join(base_dir, 'models.pkl'), 'wb') as f:
        pickle.dump(compat_dict, f)

    # Comprehensive Metadata JSON
    metadata = {
        "best_model": best_key,
        "best_model_name": best_info['name'],
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "path": "data/air_quality_data.csv",
            "total_samples": len(df_raw),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features": feature_names,
            "target": "AQI"
        },
        "features": feature_names,
        "models": models_metrics
    }

    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Copy metadata to results as well for auditing
    with open(os.path.join(results_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"      [OK] All 7 models saved to {models_dir}")
    print(f"      [OK] Results exported to {results_dir}")
    print(f"      [OK] Metadata updated at {metadata_path}")
    print("\nMulti-model training and evaluation pipeline completed successfully!\n")

    return trained_models, metadata

if __name__ == '__main__':
    run_multi_model_pipeline()
