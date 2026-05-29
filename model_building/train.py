
# ENGINE PREDICTIVE MAINTENANCE - MODEL TRAINING PIPELINE

import os
import sys
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    create_repo
)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

# MLFLOW SETUP

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("engine-maintenance-experiment")

# HUGGING FACE TOKEN

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print(" ERROR: HF_TOKEN environment variable not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# LOAD TRAIN / TEST DATA FROM HUGGING FACE

from datasets import load_dataset

print("Loading datasets from Hugging Face...")

train_ds = load_dataset(
    "MaitreyiAshok/maintenance-data-files",
    split="train_raw"
)

test_ds = load_dataset(
    "MaitreyiAshok/maintenance-data-files",
    split="test_raw"
)

# Convert to pandas
train_df = train_ds.to_pandas()

test_df = test_ds.to_pandas()

print("Train Shape:", train_df.shape)

print("Test Shape:", test_df.shape)


print("\n Downloading datasets from Hugging Face...")

# FEATURE & TARGET SPLIT

TARGET = "Engine_Condition"

FEATURES = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature"
]

Xtrain = train_df[FEATURES]

ytrain = train_df[TARGET]

Xtest = test_df[FEATURES]

ytest = test_df[TARGET]

# NUMERIC FEATURE IDENTIFICATION

numeric_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\n Numeric Features:")

print(numeric_features)

# PREPROCESSING PIPELINE

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        )
    ]
)

# MODEL PIPELINE

pipeline = Pipeline(steps=[

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    )
])

# HYPERPARAMETER GRID

param_grid = {

    "model__n_estimators": [50, 100],

    "model__max_depth": [3, 5, 7],

    "model__learning_rate": [0.01, 0.1],

    "model__subsample": [0.8, 1.0],

    "model__colsample_bytree": [0.8, 1.0]
}

# GRID SEARCH

grid = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    cv=3,

    n_jobs=-1,

    verbose=1,

    scoring="f1"
)

# TRAINING + MLFLOW LOGGING

print("\n Starting model training...")

with mlflow.start_run():

    # Train Model
    grid.fit(Xtrain, ytrain)

    results = grid.cv_results_

    # LOG ALL HYPERPARAMETER COMBINATIONS

    print("\n Logging hyperparameter combinations...")

    for i in range(len(results["params"])):

        with mlflow.start_run(nested=True):

            mlflow.log_params(
                results["params"][i]
            )

            mlflow.log_metric(
                "mean_test_score",
                results["mean_test_score"][i]
            )

            mlflow.log_metric(
                "std_test_score",
                results["std_test_score"][i]
            )

    # BEST MODEL

    best_model = grid.best_estimator_

    print("\n Best Parameters:")

    print(grid.best_params_)

    # PREDICTION

    preds = best_model.predict(Xtest)

    # EVALUATION

    acc = accuracy_score(ytest, preds)

    print(f"\n Accuracy: {acc:.4f}")

    print("\n Classification Report:\n")

    print(
        classification_report(
            ytest,
            preds
        )
    )

    print("\n Confusion Matrix:\n")

    print(
        confusion_matrix(
            ytest,
            preds
        )
    )

    # LOG BEST MODEL DETAILS

    mlflow.log_params(
        grid.best_params_
    )

    mlflow.log_metric(
        "accuracy",
        acc
    )

    # SAVE MODEL LOCALLY

    model_path = "model.pkl"

    joblib.dump(
        best_model,
        model_path
    )

    print("\n Model saved locally")

    # CREATE MODEL REPOSITORY

    repo_id_model = "MaitreyiAshok/maintenance-prediction-model"

    create_repo(
        repo_id=repo_id_model,
        repo_type="model",
        exist_ok=True
    )

    # UPLOAD MODEL TO HUGGING FACE
    print("\n Uploading model to Hugging Face...")

    api.upload_file(

        path_or_fileobj=model_path,

        path_in_repo="model.pkl",

        repo_id=repo_id_model,

        repo_type="model"
    )

    print(
        "\n Model uploaded to Hugging Face successfully!"
    )

# FINISHED

print("\n TRAINING PIPELINE COMPLETED SUCCESSFULLY!")

print(
    f"\n Model Repository:\n"
    f"https://huggingface.co/{repo_id_model}"
)
