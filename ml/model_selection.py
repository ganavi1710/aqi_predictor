"""
Model Selection and Comparative Reporting Module.
Ranks all evaluated models, automatically selects the best performer,
and exports tabular results (CSV and JSON).
"""

import os
import json
import pandas as pd

def select_best_model(models_metrics: dict) -> tuple:
    """
    Ranks models based on:
    1. Lowest RMSE (primary)
    2. Highest R2 Score (secondary)
    3. Lowest MAE (tertiary)
    Returns (best_model_key, best_model_info).
    """
    ranked = sorted(
        models_metrics.items(),
        key=lambda item: (
            item[1]['test_metrics']['RMSE'],
            -item[1]['test_metrics']['R2_Score'],
            item[1]['test_metrics']['MAE']
        )
    )
    best_key, best_info = ranked[0]
    return best_key, best_info

def export_model_comparison(models_metrics: dict, output_dir: str, best_key: str):
    """
    Exports model comparison table to CSV and JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for key, data in models_metrics.items():
        tm = data['test_metrics']
        rows.append({
            'Model_Key': key,
            'Algorithm': data['name'],
            'MAE': tm['MAE'],
            'MSE': tm['MSE'],
            'RMSE': tm['RMSE'],
            'R2_Score': tm['R2_Score'],
            'CV_Mean_RMSE': data.get('cv_mean_rmse', 'N/A'),
            'CV_Std_RMSE': data.get('cv_std_rmse', 'N/A'),
            'Training_Time_Sec': data.get('training_time_sec', 0.0),
            'Is_Best_Model': (key == best_key)
        })

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by=['RMSE', 'R2_Score'], ascending=[True, False])

    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df_sorted.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, 'model_metrics.json')
    with open(json_path, 'w') as f:
        json.dump({
            'best_model': best_key,
            'models': models_metrics,
            'summary_table': df_sorted.to_dict(orient='records')
        }, f, indent=4)

    return df_sorted

def print_model_comparison_table(df_summary: pd.DataFrame, best_key: str):
    """Prints a formatted ASCII comparison table to the terminal."""
    print("\n" + "="*85)
    print("                    MULTIPLE ALGORITHM PERFORMANCE BENCHMARK")
    print("="*85)
    header = f"{'Algorithm':<28} | {'MAE':<7} | {'RMSE':<7} | {'R2 Score':<8} | {'CV RMSE':<10} | {'Time (s)':<8}"
    print(header)
    print("-" * 85)

    for _, r in df_summary.iterrows():
        is_best = " *" if r['Model_Key'] == best_key else ""
        cv_str = f"{r['CV_Mean_RMSE']:.2f}" if isinstance(r['CV_Mean_RMSE'], (int, float)) else str(r['CV_Mean_RMSE'])
        time_str = f"{r['Training_Time_Sec']:.3f}"
        name = f"{r['Algorithm']}{is_best}"
        print(f"{name:<28} | {r['MAE']:<7.3f} | {r['RMSE']:<7.3f} | {r['R2_Score']:<8.4f} | {cv_str:<10} | {time_str:<8}")

    print("="*85)
    print("* Best model determined automatically via lowest test RMSE & highest R2.")
    print("Regression metrics should not be confused with classification accuracy.\n")
