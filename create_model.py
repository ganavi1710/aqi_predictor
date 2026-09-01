"""
Model creation entry point.
Invokes the multi-algorithm Machine Learning training, cross-validation,
and benchmarking pipeline.
"""

from ml.train_models import run_multi_model_pipeline

if __name__ == '__main__':
    run_multi_model_pipeline()
