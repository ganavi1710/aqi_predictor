# Smart Air Quality Monitoring & Prediction System 🌿

An end-to-end full-stack atmospheric monitoring and multi-algorithm machine learning platform that captures real-time global telemetry via Open-Meteo, evaluates environmental and meteorological indicators across **seven Machine Learning regression algorithms**, automatically routes predictions to the highest-performing model, categorizes severity via EPA standards, and audits predictions in a persistent SQLite database.

---

## 1. System Architecture

```
                    Historical Air Quality Dataset
                                  ↓
                   Common Preprocessing Pipeline
              (Deduplication, Range Validation, Imputation)
                                  ↓
                           Feature Matrix
         [PM2.5, CO, NO2, Temperature, Humidity, Wind Speed]
                                  ↓
                   80% Train / 20% Test Partition
                         (random_state=42)
                                  ↓
      ┌───────────────────────────┼───────────────────────────┐
      ↓                           ↓                           ↓
Linear Regression        Decision Tree Regressor    Random Forest Regressor
(Analytic Baseline)        (GridSearchCV Tuned)       (GridSearchCV Tuned)
      ↓                           ↓                           ↓
Gradient Boosting        Extra Trees Regressor       Support Vector Regressor
(GridSearchCV Tuned)       (GridSearchCV Tuned)      (StandardScaler Pipeline)
      ↓                           ↓                           ↓
                      K-Nearest Neighbors Regressor
                        (StandardScaler Pipeline)
                                  ↓
                   Same Unseen Test Set Evaluation
                     (MAE, MSE, RMSE, R² Score)
                                  ↓
                     Performance Benchmarking
                                  ↓
                Automated Best Model Selection Engine
                (Primary: Lowest RMSE; Secondary: R²)
                                  ↓
            ┌─────────────────────┴─────────────────────┐
            ↓                                           ↓
    Flask Prediction APIs                      Interactive Web Dashboard
   POST /predict, /predict-best              Comparison Charts, Residuals,
   GET /api/model-comparison                   Live Map & Telemetry Autofill
```

---

## 2. Multi-Algorithm Machine Learning Suite

Seven regression algorithms were evaluated using identical training partitions (1,200 samples) with 5-fold cross-validation, and tested against the exact same unseen test set (300 samples).

### 1. Linear Regression (Baseline)
- **Principle:** Models a linear relationship $y = \mathbf{w}^T \mathbf{x} + b$ using ordinary least squares.
- **Suitability:** Serves as a transparent linear baseline to benchmark whether atmospheric dynamics necessitate non-linear model capacity.
- **Advantages:** Fast computation, direct coefficient interpretability.
- **Limitations:** Cannot capture piecewise EPA breakpoints or complex meteorological interaction effects without manual polynomial expansions.
- **Evaluation:** MAE: `12.164` | RMSE: `16.995` | $R^2$: `0.8663`

### 2. Decision Tree Regressor
- **Principle:** Recursively partitions the feature space into orthogonal regions by minimizing variance (Friedman MSE criterion).
- **Suitability:** Naturally models the piecewise step-functions inherent in EPA breakpoint calculations.
- **Advantages:** Completely scale-invariant, intuitive decision logic, handles non-linear splits.
- **Limitations:** High variance; prone to overfitting on continuous functions when unpruned.
- **Hyperparameters Tuned:** `max_depth=10`, `min_samples_split=2`, `min_samples_leaf=4`, `criterion='friedman_mse'`.
- **Evaluation:** MAE: `3.387` | RMSE: `4.854` | $R^2$: `0.9891`

### 3. Random Forest Regressor
- **Principle:** Bootstrap aggregation (bagging) of decorrelated decision trees, averaging their predictions to reduce variance.
- **Suitability:** Smooths out the abrupt boundaries of single trees while handling complex environmental interactions.
- **Advantages:** Resistant to overfitting, provides robust out-of-bag estimates and feature importances.
- **Limitations:** Higher memory footprint than single trees.
- **Hyperparameters Tuned:** `n_estimators=100`, `max_depth=12`, `min_samples_split=2`, `min_samples_leaf=1`.
- **Evaluation:** MAE: `2.732` | RMSE: `3.684` | $R^2$: `0.9937`

### 4. Gradient Boosting Regressor (🏆 Best Performing Model)
- **Principle:** Sequentially builds shallow trees, each trained to predict the negative gradient (pseudo-residuals) of the loss function.
- **Suitability:** Exceptional at refining predictions around critical EPA transition boundaries (e.g. Moderate to Unhealthy).
- **Advantages:** Consistently achieves the lowest residual error on non-linear atmospheric data.
- **Limitations:** Sequential training makes it computationally slower to train than bagging.
- **Hyperparameters Tuned:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `min_samples_split=4`.
- **Evaluation:** MAE: `2.350` | RMSE: `3.144` | $R^2$: `0.9954`

### 5. Extra Trees Regressor (Extremely Randomized Trees)
- **Principle:** Ensembles randomized decision trees where cut-points are chosen at random rather than optimal thresholds.
- **Suitability:** Provides strong regularization against noisy atmospheric sensor packets.
- **Advantages:** Faster training than standard Random Forest with comparable or superior variance reduction.
- **Limitations:** Random splits can occasionally miss subtle boundary thresholds.
- **Hyperparameters Tuned:** `n_estimators=100`, `max_depth=12`, `min_samples_split=2`, `min_samples_leaf=1`.
- **Evaluation:** MAE: `2.635` | RMSE: `3.341` | $R^2$: `0.9948`

### 6. Support Vector Regression (SVR - RBF Kernel)
- **Principle:** Finds a continuous function that deviates from target values by no more than $\varepsilon$, with $L_2$ regularization in an infinite-dimensional reproducing kernel Hilbert space (RKHS).
- **Suitability:** Models smooth, continuous non-linear manifold relationships.
- **Data Leakage Safeguard:** Wrapped inside an `sklearn.pipeline.Pipeline` with `StandardScaler` fitted strictly on training folds.
- **Advantages:** Robust to outliers lying within the $\varepsilon$-insensitive tube.
- **Limitations:** Requires feature scaling; sensitive to hyperparameter tuning ($C, \varepsilon$).
- **Hyperparameters Tuned:** `C=300.0`, `epsilon=0.1`, `kernel='rbf'`.
- **Evaluation:** MAE: `4.171` | RMSE: `6.530` | $R^2$: `0.9803`

### 7. K-Nearest Neighbors Regressor (KNN)
- **Principle:** Predicts AQI based on distance-weighted local averaging of the $k$ nearest historical environmental observations.
- **Suitability:** Non-parametric reference model reflecting local historical precedent.
- **Data Leakage Safeguard:** Wrapped inside an `sklearn.pipeline.Pipeline` with `StandardScaler` fitted strictly on training folds.
- **Advantages:** No explicit training phase, highly intuitive local clustering.
- **Limitations:** High inference latency on large datasets; degraded performance in high-dimensional feature spaces.
- **Hyperparameters Tuned:** `n_neighbors=5`, `weights='distance'`.
- **Evaluation:** MAE: `10.431` | RMSE: `13.527` | $R^2$: `0.9153`

---

## 3. Algorithm Benchmark & Performance Comparison

Evaluated on the **same 20% unseen test partition (300 records)**:

| Rank | Algorithm | Test RMSE ↓ | Test MAE ↓ | Test MSE | $R^2$ Score ↑ | 5-Fold CV RMSE | Train Time | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Gradient Boosting Regressor** | **3.144** | **2.350** | **9.887** | **0.9954** | **6.142 (±1.64)** | 5.37s | **Winner (Selected)** |
| 🥈 | **Extra Trees Regressor** | 3.341 | 2.635 | 11.163 | 0.9948 | 6.316 (±0.27) | 2.91s | Evaluated |
| 🥉 | **Random Forest Regressor** | 3.684 | 2.732 | 13.573 | 0.9937 | 6.502 (±0.09) | 5.44s | Evaluated |
| 4 | **Decision Tree Regressor** | 4.854 | 3.387 | 23.557 | 0.9891 | 7.156 (±0.18) | 2.48s | Evaluated |
| 5 | **Support Vector Regression (SVR)** | 6.530 | 4.171 | 42.638 | 0.9803 | 10.201 (±2.32) | 2.21s | Evaluated |
| 6 | **K-Nearest Neighbors (KNN)** | 13.527 | 10.431 | 182.985 | 0.9153 | 15.847 (±0.37) | 0.15s | Evaluated |
| 7 | **Linear Regression** | 16.995 | 12.164 | 288.816 | 0.8663 | 17.194 (±5.48) | 0.01s | Baseline |

> **Note on Metrics:** $R^2$ represents the coefficient of determination (the proportion of AQI variance explained by the model) and is explicitly distinct from classification accuracy.

---

## 4. How the Best Model Was Selected

The model selection engine (`ml/model_selection.py`) ranks all candidate models dynamically using the following hierarchical criteria:
1. **Primary Criterion:** Lowest Test Root Mean Squared Error (RMSE), penalizing large outlier errors heavily.
2. **Secondary Criterion:** Highest Coefficient of Determination ($R^2$).
3. **Tertiary Criterion:** Lowest Test Mean Absolute Error (MAE).

**Result:** **Gradient Boosting Regressor** achieved the lowest RMSE ($3.144$) and highest $R^2$ ($0.9954$). The selection is fully data-driven and loaded dynamically from `models/model_metadata.json` without hard-coding.

---

## 5. Environmental Feature Importances

Calculated across the tree-based ensembles via Gini / variance reduction:

| Feature | Gradient Boosting | Extra Trees | Random Forest | Decision Tree | Interpretation |
|---|:---:|:---:|:---:|:---:|---|
| **PM2.5** | **94.82%** | **90.41%** | **94.28%** | **96.10%** | Dominant inhalable particulate driver |
| **CO** | **4.21%** | **7.12%** | **4.61%** | **3.56%** | Vehicular combustion indicator |
| **Wind Speed** | **0.54%** | **1.14%** | **0.55%** | **0.21%** | Atmospheric dispersion factor |
| **Humidity** | **0.22%** | **0.65%** | **0.31%** | **0.07%** | Particle trapping and inversion |
| **NO2** | **0.15%** | **0.48%** | **0.19%** | **0.05%** | Nitrogen oxide emission factor |
| **Temperature** | **0.06%** | **0.20%** | **0.06%** | **0.01%** | Thermal boundary layer driver |

---

## 6. Prediction API Specification

### `POST /predict`
Accepts environmental measurements with an optional algorithm selector:

```json
{
    "model": "gradient_boosting",
    "pm25": 85.0,
    "co": 1.25,
    "no2": 42.0,
    "temperature": 28.5,
    "humidity": 65.0,
    "wind_speed": 4.2,
    "location": "Central Station"
}
```
*Supported `model` values:* `"best_model"` *(default)*, `"gradient_boosting"`, `"extra_trees"`, `"random_forest"`, `"decision_tree"`, `"svr"`, `"knn"`, `"linear_regression"`.

**Response (`200 OK`):**
```json
{
    "success": true,
    "id": 1,
    "predicted_aqi": 138.25,
    "category": "Unhealthy for Sensitive Groups",
    "badge": "Unhealthy for Sensitive Groups 😷",
    "color": "#ff9900",
    "recommendation": "Sensitive individuals should reduce prolonged outdoor activity.",
    "tips": [
        "Children, elderly, and individuals with respiratory issues should limit time outside",
        "Wear an N95 mask during prolonged exposure",
        "Keep indoor air filtered and clean"
    ],
    "model": "Gradient Boosting Regressor",
    "model_key": "gradient_boosting",
    "is_best_model": true,
    "features_used": ["PM2.5", "CO", "NO2", "Temperature", "Humidity", "Wind_Speed"],
    "inputs": {
        "PM2.5": 85.0,
        "CO": 1.25,
        "NO2": 42.0,
        "Temperature": 28.5,
        "Humidity": 65.0,
        "Wind_Speed": 4.2
    }
}
```

### `POST /predict-best`
Shortcut endpoint that automatically evaluates input against the winning model.

### `GET /api/model-comparison`
Returns metadata, test metrics, cross-validation scores, and feature importances for all 7 models.

### `GET /api/history`
Returns historical predictions logged in the SQLite audit database.

---

## 7. How to Train & Benchmark Models

To retrain all 7 models, run 5-fold cross-validation, and regenerate comparison charts:
```powershell
python ml/train_models.py
```
*(Or `python create_model.py`)*

Outputs generated:
- Models: `models/*.pkl` (all 7 models + backward-compatible wrappers)
- Diagnostic Plots: `models/evaluation_plots/*.png` (Actual vs Predicted, Residuals, Feature Importances)
- CSV Comparison: `results/model_comparison.csv`
- JSON Metrics: `results/model_metrics.json`

---

## 8. How to Run Tests & Server

### Run Unit Tests:
```powershell
python -m unittest tests/test_multi_model_pipeline.py -v
```
*(Tests model loading, inference for all 7 algorithms, best model routing, input validation, and SQLite persistence)*

### Launch the Flask Dashboard:
```powershell
python app.py
```
Open **`http://localhost:5001`** in your browser.

---

## 9. Known Limitations

1. **Microclimate Turbulence:** Localized street canyons and industrial exhaust plumes can create micro-spikes that regional weather stations may smooth over.
2. **Additional Photochemical Pollutants:** Incorporating ozone ($O_3$) and sulfur dioxide ($SO_2$) will provide even deeper coverage of photochemical smog events.
