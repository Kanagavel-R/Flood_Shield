# ==========================================================
# Flood Shield
# Module 1 : Data Collection & Data Preprocessing
# ==========================================================

# ==========================
# STEP 1 : Import Libraries
# ==========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

print("=" * 60)
print("FLOOD SHIELD - MODULE 1")
print("=" * 60)

# ==========================================================
# STEP 2 : Create Required Folders
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed")
GRAPH_FOLDER = os.path.join(BASE_DIR, "output", "graphs")

os.makedirs(PROCESSED_DATA, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# ==========================================================
# STEP 3 : Load Dataset
# ==========================================================

DATA_PATH = os.path.join(RAW_DATA, "flood.csv")

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully\n")

# ==========================================================
# STEP 4 : Display Dataset
# ==========================================================

print("="*60)
print("FIRST FIVE RECORDS")
print("="*60)

print(df.head())

print("\n")

print("="*60)
print("LAST FIVE RECORDS")
print("="*60)

print(df.tail())

# ==========================================================
# STEP 5 : Dataset Shape
# ==========================================================

print("\n")

print("="*60)
print("DATASET SHAPE")
print("="*60)

print(df.shape)

# ==========================================================
# STEP 6 : Column Names
# ==========================================================

print("\n")

print("="*60)
print("COLUMN NAMES")
print("="*60)

print(df.columns)

# ==========================================================
# STEP 7 : Dataset Information
# ==========================================================

print("\n")

print("="*60)
print("DATASET INFORMATION")
print("="*60)

print(df.info())

# ==========================================================
# STEP 8 : Statistical Summary
# ==========================================================

print("\n")

print("="*60)
print("STATISTICAL SUMMARY")
print("="*60)

print(df.describe())

# ==========================================================
# STEP 9 : Missing Values
# ==========================================================

print("\n")

print("="*60)
print("MISSING VALUES")
print("="*60)

print(df.isnull().sum())

# ==========================================================
# STEP 10 : Duplicate Values
# ==========================================================

duplicates = df.duplicated().sum()

print("\n")

print("="*60)
print("DUPLICATE ROWS")
print("="*60)

print(duplicates)

if duplicates > 0:

    df.drop_duplicates(inplace=True)

    print("Duplicates Removed")

else:

    print("No Duplicate Rows Found")

# ==========================================================
# STEP 11 : Correlation Heatmap
# ==========================================================

plt.figure(figsize=(14,10))

plt.imshow(df.corr(), cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(df.columns)), df.columns, rotation=90)

plt.yticks(range(len(df.columns)), df.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(os.path.join(
    GRAPH_FOLDER,
    "correlation_heatmap.png"
))

plt.close()

print("\nCorrelation Heatmap Saved")

# ==========================================================
# STEP 12 : Histogram
# ==========================================================

df.hist(figsize=(16,12))

plt.tight_layout()

plt.savefig(os.path.join(
    GRAPH_FOLDER,
    "histogram.png"
))

plt.close()

print("Histogram Saved")

# ==========================================================
# STEP 13 : Box Plot
# ==========================================================

plt.figure(figsize=(18,8))

df.boxplot(rot=90)

plt.tight_layout()

plt.savefig(os.path.join(
    GRAPH_FOLDER,
    "boxplot.png"
))

plt.close()

print("Box Plot Saved")

# ==========================================================
# STEP 14 : Separate Features and Target
# ==========================================================

X = df.drop("FloodProbability", axis=1)

y = df["FloodProbability"]

print("\n")

print("="*60)
print("FEATURES AND TARGET")
print("="*60)

print("Features Shape :", X.shape)

print("Target Shape :", y.shape)

# ==========================================================
# STEP 15 : Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed")

# ==========================================================
# STEP 16 : Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.20,

    random_state=42

)

print("\n")

print("="*60)
print("TRAIN TEST SPLIT")
print("="*60)

print("Training Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)

# ==========================================================
# STEP 17 : Cross Validation
# ==========================================================

model = LinearRegression()

scores = cross_val_score(

    model,

    X_scaled,

    y,

    cv=5,

    scoring="r2"

)

print("\n")

print("="*60)
print("5-FOLD CROSS VALIDATION")
print("="*60)

print(scores)

print()

print("Average R2 Score :", scores.mean())

# ==========================================================
# STEP 18 : Save Processed Dataset
# ==========================================================

processed = pd.DataFrame(

    X_scaled,

    columns=X.columns

)

processed["FloodProbability"] = y.values

processed.to_csv(

    os.path.join(
        PROCESSED_DATA,
        "processed_data.csv"
    ),

    index=False

)

print("\nProcessed Dataset Saved")

# ==========================================================
# STEP 19 : Module Summary
# ==========================================================

print("\n")

print("="*60)
print("MODULE 1 SUMMARY")
print("="*60)

print("Dataset Shape :", df.shape)

print("Number of Features :", X.shape[1])

print("Target :", "FloodProbability")

print("Missing Values :", df.isnull().sum().sum())

print("Duplicate Rows :", duplicates)

print("Feature Scaling : Completed")

print("Train-Test Split : Completed")

print("Cross Validation : Completed")

print("Processed Dataset : Saved")

print("Graphs Generated :")
print("1. Correlation Heatmap")
print("2. Histogram")
print("3. Box Plot")

print("\nMODULE 1 COMPLETED SUCCESSFULLY")

print("="*60)