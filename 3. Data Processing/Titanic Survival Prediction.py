# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC Titanic Survival Prediction — Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC Step 1 — Load and inspect the data

# COMMAND ----------

import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2 — Check data quality

# COMMAND ----------

print("Dataset rows and columns:", df.shape)

print("\nMissing values:")
display(df.isnull().sum().sort_values(ascending=False))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
display(df["survived"].value_counts())

# COMMAND ----------

# Inspect the dataset

df.info()
display(df.describe(include="all").T)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 3 — Choose the features and target

# COMMAND ----------


#We need to separate the information used to make predictions from the answer we want the model to learn
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

target = "survived"

X = df[features].copy()
y = df[target].copy()

print("Input features:", X.columns.tolist())
print("Target:", target)

display(X.head())

#X contains passenger information the model can use.
#y contains the actual survival outcome.
#We leave out columns such as alive, because they directly reveal the target and would give the model an unfair advantage.

# COMMAND ----------

# MAGIC %md
# MAGIC Step 4 — Split the data into training and testing sets

# COMMAND ----------

# DBTITLE 1,Testing

#We will use 80% of the data for training and keep 20% separate for testing.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

#The model learns from the training set. The test set helps us measure how well it predicts outcomes for passengers it did not train on. stratify=y helps preserve the survival-class proportions in both sets.

# COMMAND ----------

# MAGIC %md
# MAGIC Step 5 — Prepare missing values and categorical data

# COMMAND ----------

# DBTITLE 1,Algorithms
#Machine learning algorithms generally need missing values handled and text categories converted into numerical representations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

numeric_features = [
    "pclass", "age", "sibsp", "parch", "fare"
]

categorical_features = ["sex", "embarked"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("Preprocessing steps are ready.")

# Fit and transform the training data to preview the preprocessed output
X_train_processed = preprocessor.fit_transform(X_train)
processed_columns = preprocessor.get_feature_names_out()
X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_columns)

print("Processed feature names:", processed_columns.tolist())
print("Processed training data shape:", X_train_processed_df.shape)
display(X_train_processed_df.head(10))

#Fills missing numeric values using the median.
#Fills missing text categories using the most frequent value.
#Converts categories such as male and female into numerical indicators.
#Putting preprocessing into a pipeline helps prevent information from the test set leaking into the training process

# COMMAND ----------

# MAGIC %md
# MAGIC Step 6 — Train the machine learning model

# COMMAND ----------

# DBTITLE 1,training an actual model
from sklearn.ensemble import RandomForestClassifier

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

print("Model training completed.")

#RandomForestClassifier creates the prediction model.
#n_estimators=200 uses 200 decision trees.
#random_state=42 helps make results reproducible.
#fit() trains the model using the training data.

# COMMAND ----------

# MAGIC %md
# MAGIC Step 7 — Make predictions on the test set

# COMMAND ----------

y_pred = model.predict(X_test)

results = X_test.copy()
results["actual_survived"] = y_test
results["predicted_survived"] = y_pred

display(results.head(10))

#The predicted_survived column contains the model's predictions:
# 1 = predicted to have survived.
# 0 = predicted not to have survived.

# COMMAND ----------

# MAGIC %md
# MAGIC Step 8 — Evaluate model performance

# COMMAND ----------


#A trained model is not automatically a good model. We must evaluate its predictions using several metrics.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

print("\nDetailed classification report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Did not survive", "Survived"],
    zero_division=0
))

#Accuracy	Percentage of all test predictions that were correct.
#Precision	Of those predicted to have survived, how many actually survived?
#Recall	Of those who actually survived, how many did the model identify?
#F1-score	A combined measure of precision and recall.

# COMMAND ----------

# MAGIC %md
# MAGIC Step 9 — Examine the confusion matrix

# COMMAND ----------

import pandas as pd
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual: Did not survive", "Actual: Survived"],
    columns=["Predicted: Did not survive", "Predicted: Survived"]
)

display(cm_df)

#Top-left: Correctly predicted non-survivors.
#Top-right: Non-survivors incorrectly predicted to have survived.
#Bottom-left: Survivors incorrectly predicted not to have survived.
#Bottom-right: Correctly predicted survivors

# COMMAND ----------

# MAGIC %md
# MAGIC Step 10 — Identify influential features

# COMMAND ----------

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
importances = fitted_classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

display(importance_df.head(10))

# COMMAND ----------

import matplotlib.pyplot as plt

top_features = importance_df.head(10).sort_values("importance")

plt.figure(figsize=(9, 5))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Used by the Titanic Model")
plt.tight_layout()
plt.show()

#Feature importance shows how much the model relied on a feature. It does not prove that the feature caused survival, and correlated features can affect how importance is distributed

# COMMAND ----------

# MAGIC %md
# MAGIC Step 11 — Predict survival probabilities

# COMMAND ----------

probabilities = model.predict_proba(X_test)

results["probability_survived"] = probabilities[:, 1]

display(
    results[
        ["actual_survived", "predicted_survived", "probability_survived"]
    ].head(10)
)

#a probability of 0.80 means the model estimates an 80% chance of survival for that passenger, based on the patterns it learned. It is not a guarantee that the passenger survived.

# COMMAND ----------

# MAGIC %md
# MAGIC Step 12 — Test a new passenger

# COMMAND ----------

#Create a hypothetical passenger record and ask the trained model for a prediction.

new_passenger = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 80.0,
    "embarked": "S"
}])

prediction = model.predict(new_passenger)[0]
survival_probability = model.predict_proba(new_passenger)[0, 1]

print(
    "Prediction:",
    "Survived" if prediction == 1 else "Did not survive"
)
print(f"Estimated survival probability: {survival_probability:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC Step 13 — Save your trained model

# COMMAND ----------

import joblib

model_path = "/tmp/titanic_random_forest_model.joblib"

joblib.dump(model, model_path)

print("Model saved to:", model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 14 — Findings