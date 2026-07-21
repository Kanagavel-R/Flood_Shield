# ==========================================================
# Flood Shield
# Module 2 : Supervised Learning
# ==========================================================

# ==========================================================
# STEP 1 : Import Libraries
# ==========================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("="*60)
print("FLOOD SHIELD - MODULE 2")
print("="*60)

# ==========================================================
# STEP 2 : Load Processed Dataset
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "processed_data.csv"
)

df = pd.read_csv(DATA_PATH)

print("\nProcessed Dataset Loaded Successfully\n")

# ==========================================================
# STEP 3 : Create Flood Risk Classes
# ==========================================================

df["FloodRisk"] = pd.qcut(
    df["FloodProbability"],
    q=3,
    labels=[0,1,2]
)

print("="*60)
print("CLASS DISTRIBUTION")
print("="*60)

print(df["FloodRisk"].value_counts())

# ==========================================================
# STEP 4 : Features & Target
# ==========================================================

X = df.drop(
    ["FloodProbability","FloodRisk"],
    axis=1
)

y = df["FloodRisk"]

print("\nFeatures :",X.shape)

print("Target :",y.shape)

# ==========================================================
# STEP 5 : Train Test Split
# ==========================================================

X_train,X_test,y_train,y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTrain Shape :",X_train.shape)

print("Test Shape :",X_test.shape)

# ==========================================================
# STEP 6 : Model Evaluation Function
# ==========================================================

results=[]

trained_models={}

def evaluate(model,name):

    model.fit(
        X_train,
        y_train
    )

    prediction=model.predict(X_test)

    accuracy=accuracy_score(
        y_test,
        prediction
    )

    precision=precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall=recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1=f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    print("\n"+"="*60)

    print(name)

    print("="*60)

    print("Accuracy :",accuracy)

    print("Precision :",precision)

    print("Recall :",recall)

    print("F1 Score :",f1)

    print("\nClassification Report\n")

    print(classification_report(
        y_test,
        prediction,
        zero_division=0
    ))

    results.append([

        name,

        accuracy,

        precision,

        recall,

        f1

    ])

    trained_models[name]=model

    return prediction

# ==========================================================
# STEP 7 : Logistic Regression
# ==========================================================

log_model=LogisticRegression(
    max_iter=1000
)

log_prediction=evaluate(
    log_model,
    "Logistic Regression"
)

# ==========================================================
# STEP 8 : Decision Tree
# ==========================================================

dt_model=DecisionTreeClassifier(
    random_state=42
)

dt_prediction=evaluate(
    dt_model,
    "Decision Tree"
)

# ==========================================================
# STEP 9 : KNN
# ==========================================================

knn_model=KNeighborsClassifier(
    n_neighbors=5
)

knn_prediction=evaluate(
    knn_model,
    "KNN"
)

# ==========================================================
# STEP 10 : Random Forest
# ==========================================================

rf_model=RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

rf_prediction=evaluate(

    rf_model,

    "Random Forest"

)

print("\n")

print("="*60)

print("PART 1 COMPLETED")

print("="*60)

# ==========================================================
# STEP 11 : Confusion Matrix
# ==========================================================

print("\n")
print("="*60)
print("CONFUSION MATRICES")
print("="*60)

predictions = {
    "Logistic Regression": log_prediction,
    "Decision Tree": dt_prediction,
    "KNN": knn_prediction,
    "Random Forest": rf_prediction
}

for name, pred in predictions.items():

    print(f"\n{name}")

    cm = confusion_matrix(y_test, pred)

    print(cm)

# ==========================================================
# STEP 12 : Confusion Matrix Heatmaps
# ==========================================================

GRAPH_DIR = os.path.join(
    BASE_DIR,
    "output",
    "graphs"
)

os.makedirs(GRAPH_DIR, exist_ok=True)

for name, pred in predictions.items():

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6,5))

    plt.imshow(cm, cmap="Blues")

    plt.title(name)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_cm.png"

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            filename
        )
    )

    plt.close()

print("\nConfusion Matrix Images Saved")

# ==========================================================
# STEP 13 : Model Comparison Table
# ==========================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Algorithm",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(results_df)

# ==========================================================
# STEP 14 : Save Model Comparison Report
# ==========================================================

REPORT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)

results_df.to_csv(

    os.path.join(
        REPORT_DIR,
        "model_comparison.csv"
    ),

    index=False

)

print("\nModel Comparison Report Saved")

# ==========================================================
# STEP 15 : Accuracy Comparison Graph
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(

    results_df["Algorithm"],

    results_df["Accuracy"]

)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    os.path.join(

        GRAPH_DIR,

        "accuracy_comparison.png"

    )

)

plt.close()

print("Accuracy Comparison Graph Saved")

# ==========================================================
# STEP 16 : Feature Importance (Random Forest)
# ==========================================================

importance = rf_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("="*60)
print("TOP 10 IMPORTANT FEATURES")
print("="*60)

print(importance_df.head(10))

plt.figure(figsize=(10,6))

plt.barh(

    importance_df["Feature"][:10],

    importance_df["Importance"][:10]

)

plt.gca().invert_yaxis()

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig(

    os.path.join(

        GRAPH_DIR,

        "feature_importance.png"

    )

)

plt.close()

print("Feature Importance Graph Saved")

print("\n")
print("="*60)
print("PART 2 COMPLETED")
print("="*60)

# ==========================================================
# STEP 17 : Hyperparameter Tuning (GridSearchCV)
# ==========================================================

from sklearn.model_selection import GridSearchCV

print("\n")
print("="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

param_grid = {

    "n_estimators":[50,100,150],

    "max_depth":[10,20,None],

    "min_samples_split":[2,5]

}

grid = GridSearchCV(

    RandomForestClassifier(random_state=42),

    param_grid=param_grid,

    cv=3,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(X_train,y_train)

best_rf = grid.best_estimator_

print("\nBest Parameters")

print(grid.best_params_)

print("\nBest Cross Validation Score")

print(grid.best_score_)

# ==========================================================
# STEP 18 : Evaluate Best Random Forest
# ==========================================================

best_prediction = best_rf.predict(X_test)

best_accuracy = accuracy_score(

    y_test,

    best_prediction

)

print("\nBest Random Forest Accuracy")

print(best_accuracy)

# ==========================================================
# STEP 19 : Save Best Model
# ==========================================================

MODEL_DIR = os.path.join(

    BASE_DIR,

    "models"

)

os.makedirs(

    MODEL_DIR,

    exist_ok=True

)

MODEL_PATH = os.path.join(

    MODEL_DIR,

    "random_forest.pkl"

)

joblib.dump(

    best_rf,

    MODEL_PATH

)

print("\nBest Model Saved Successfully")

print(MODEL_PATH)

# ==========================================================
# STEP 20 : Save Feature Importance CSV
# ==========================================================

importance_df.to_csv(

    os.path.join(

        REPORT_DIR,

        "feature_importance.csv"

    ),

    index=False

)

print("Feature Importance Report Saved")

# ==========================================================
# STEP 21 : Save Classification Report
# ==========================================================

report = classification_report(

    y_test,

    best_prediction,

    zero_division=0

)

with open(

    os.path.join(

        REPORT_DIR,

        "classification_report.txt"

    ),

    "w"

) as file:

    file.write(report)

print("Classification Report Saved")

# ==========================================================
# STEP 22 : Best Model Selection
# ==========================================================

best_algorithm = results_df.loc[

    results_df["Accuracy"].idxmax()

]

print("\n")
print("="*60)
print("BEST MODEL")
print("="*60)

print(best_algorithm)

# ==========================================================
# STEP 23 : Final Project Summary
# ==========================================================

print("\n")
print("="*60)
print("MODULE 2 SUMMARY")
print("="*60)

print("Dataset Used")

print("Flood Prediction Dataset")

print("\nAlgorithms Implemented")

print("1. Logistic Regression")

print("2. Decision Tree")

print("3. KNN")

print("4. Random Forest")

print("\nEvaluation Metrics")

print("- Accuracy")

print("- Precision")

print("- Recall")

print("- F1 Score")

print("- Confusion Matrix")

print("- Classification Report")

print("\nGraphs Generated")

print("- Accuracy Comparison")

print("- Feature Importance")

print("- Confusion Matrix Heatmaps")

print("\nReports Generated")

print("- Model Comparison CSV")

print("- Feature Importance CSV")

print("- Classification Report TXT")

print("\nSaved Model")

print("random_forest.pkl")

print("\nHyperparameter Tuning")

print("Completed using GridSearchCV")

print("\nMODULE 2 COMPLETED SUCCESSFULLY")

print("="*60)