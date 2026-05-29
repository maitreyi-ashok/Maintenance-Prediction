
# Engine Predictive Maintenance - Data Preparation Pipeline

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets import Dataset
from huggingface_hub import login, create_repo

# Configuration

DATA_PATH = os.getenv(
    "DATA_PATH",
    "maintenance_project/data/engine_data.csv"
)

HF_TOKEN = os.getenv("HF_TOKEN")

REPO_ID = "MaitreyiAshok/maintenance-data-files"

TARGET = "Engine_Condition"

FEATURES = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Validate Hugging Face Token

if not HF_TOKEN:
    print(" ERROR: HF_TOKEN environment variable not set")
    sys.exit(1)

# Login to Hugging Face
login(token=HF_TOKEN)

# Create dataset repository if not exists
create_repo(
    repo_id=REPO_ID,
    token=HF_TOKEN,
    repo_type="dataset",
    exist_ok=True
)

# Load Dataset

print("\n Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f" Original Shape: {df.shape}")

# Rename Columns (Standardized Format)

rename_map = {
    "Engine rpm": "Engine_RPM",
    "Lub oil pressure": "Lub_Oil_Pressure",
    "Fuel pressure": "Fuel_Pressure",
    "Coolant pressure": "Coolant_Pressure",
    "lub oil temp": "Lub_Oil_Temperature",
    "Coolant temp": "Coolant_Temperature",
    "Engine Condition": "Engine_Condition"
}

df.rename(columns=rename_map, inplace=True)

# Drop Unnecessary Columns

cols_to_drop = [
    c for c in df.columns
    if c.lower() in ["unnamed: 0", "id", "index", "row_id"]
]

if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f" Dropped columns: {cols_to_drop}")
else:
    print(" No unnecessary columns found")

print(f" Shape after cleanup: {df.shape}")

# Initial Inspection

print("\n Columns:")
print(df.columns.tolist())

print("\n Missing Values:")
print(df.isnull().sum())

print("\n Duplicate Rows:")
print(df.duplicated().sum())

print("\n Data Types:")
print(df.dtypes)

print("\n Statistical Summary:")
print(df.describe())

# Missing Value Handling

print("\n🔧 Handling missing values...")

missing_pct = df.isnull().mean() * 100

for col in df.columns:

    missing = df[col].isnull().sum()

    if missing > 0:

        pct = missing_pct[col]

        # Drop columns with >50% missing
        if pct > 50:

            df.drop(columns=[col], inplace=True)

            print(f" Dropped '{col}' ({pct:.1f}% missing)")

        # Numeric columns
        elif df[col].dtype in [
            np.float64,
            np.float32,
            np.int64,
            np.int32
        ]:

            median_val = df[col].median()

            df[col].fillna(median_val, inplace=True)

            print(
                f" Imputed '{col}' with median={median_val:.3f}"
            )

        # Categorical columns
        else:

            mode_val = df[col].mode()[0]

            df[col].fillna(mode_val, inplace=True)

            print(
                f" Imputed '{col}' with mode='{mode_val}'"
            )

print(
    f"\n Total Missing Values After Cleaning: "
    f"{df.isnull().sum().sum()}"
)

# Outlier Handling using IQR Capping

print("\n Handling outliers using IQR capping...")


def cap_outliers_iqr(dataframe, columns, factor=1.5):

    df_out = dataframe.copy()

    report = []

    for col in columns:

        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        n_outliers = (
            (df_out[col] < lower) |
            (df_out[col] > upper)
        ).sum()

        df_out[col] = df_out[col].clip(
            lower=lower,
            upper=upper
        )

        report.append({
            "Feature": col,
            "Lower_Bound": round(lower, 3),
            "Upper_Bound": round(upper, 3),
            "Outliers_Capped": int(n_outliers)
        })

    return df_out, pd.DataFrame(report)


numeric_features = [
    c for c in FEATURES if c in df.columns
]

df, outlier_report = cap_outliers_iqr(
    df,
    numeric_features
)

print("\n Outlier Capping Report:")
print(outlier_report.to_string(index=False))

# Remove Duplicates

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print(f"\n Removed {before - after} duplicate rows")

print(f" Shape after deduplication: {df.shape}")

# Ensure Correct Data Types

df[TARGET] = df[TARGET].astype(int)

for col in numeric_features:
    df[col] = df[col].astype(float)

print("\n Final Data Types:")
print(df.dtypes)

# Target Distribution

print("\n Target Distribution:")
print(df[TARGET].value_counts())

print("\n Target Percentage:")
print(df[TARGET].value_counts(normalize=True) * 100)

# Final Dataset Summary

print("\n Cleaned Dataset Summary:")
print(df[numeric_features + [TARGET]].describe())

# Visualize Cleaned Distributions

print("\n Generating feature distribution plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes = axes.flatten()

for i, col in enumerate(numeric_features):

    sns.histplot(
        data=df,
        x=col,
        hue=TARGET,
        ax=axes[i],
        bins=30,
        kde=True,
        alpha=0.7
    )

    axes[i].set_title(f"{col} Distribution")

plt.suptitle(
    "Feature Distributions After Cleaning",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "fig_cleaned_distributions.png",
    bbox_inches="tight"
)

plt.show()

# Train-Test Split

print("\n Splitting dataset...")

X = df[numeric_features]

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n Split Complete")

print(
    f"Train Shape: {X_train.shape} | "
    f"Failure %: {y_train.mean()*100:.2f}"
)

print(
    f"Test Shape: {X_test.shape} | "
    f"Failure %: {y_test.mean()*100:.2f}"
)

# Feature Scaling

print("\n📏 Applying Standard Scaling...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=numeric_features,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=numeric_features,
    index=X_test.index
)

print("\n Scaler Parameters:")

for col, mean, std in zip(
    numeric_features,
    scaler.mean_,
    scaler.scale_
):
    print(
        f"{col:25s} | "
        f"mean={mean:.3f} | std={std:.3f}"
    )

# Assemble Final Train/Test Sets

train_df = X_train_scaled.copy()
train_df[TARGET] = y_train.values

test_df = X_test_scaled.copy()
test_df[TARGET] = y_test.values

# Raw versions (for tree models)

train_raw_df = X_train.copy()
train_raw_df[TARGET] = y_train.values

test_raw_df = X_test.copy()
test_raw_df[TARGET] = y_test.values

print(f"\n Train Set Shape: {train_df.shape}")
print(f"\n Test Set Shape: {test_df.shape}")

# Save Locally

print("\n Saving datasets locally...")

train_df.to_csv("train_scaled.csv", index=False)
test_df.to_csv("test_scaled.csv", index=False)

train_raw_df.to_csv("train_raw.csv", index=False)
test_raw_df.to_csv("test_raw.csv", index=False)

print(" Files saved successfully")

# Upload to Hugging Face

print("\n Uploading datasets to Hugging Face...")


def upload_split(dataframe, repo_id, split_name):

    hf_ds = Dataset.from_pandas(
        dataframe.reset_index(drop=True)
    )

    hf_ds.push_to_hub(
        repo_id,
        split=split_name
    )

    print(f" Uploaded split='{split_name}'")


upload_split(train_df, REPO_ID, "train_scaled")
upload_split(test_df, REPO_ID, "test_scaled")

upload_split(train_raw_df, REPO_ID, "train_raw")
upload_split(test_raw_df, REPO_ID, "test_raw")

# Finished

print("\n ALL DATASETS UPLOADED SUCCESSFULLY!")

print(
    f"\n Dataset URL:\n"
    f"https://huggingface.co/datasets/{REPO_ID}"
)
