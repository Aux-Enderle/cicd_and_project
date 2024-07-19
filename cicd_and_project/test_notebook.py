# Databricks notebook source
from databricks.connect import DatabricksSession
import sys
import os

def isRunningInDatabricks(): 
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

def display(df):
    if isRunningInDatabricks():
        return df.display()
    else:
        return df.show()

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
dbutils.widgets.text(
    "model_name", "dev.cicd_and_project.cicd_and_project-model", label="Full (Three-Level) Model Name"
)

model_name = "auxmoney_databricks_poc.cicd_and_project.cicd_and_project-model"

# COMMAND ----------

import mlflow

mlflow.login() if not isRunningInDatabricks() else None

mlflow.set_experiment("/Users/enderle@auxmoney.com/local_test")
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

training_df = spark.read.table("samples.nyctaxi.trips")
display(training_df)

# COMMAND ----------

from mlflow.tracking import MlflowClient
import mlflow.pyfunc

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

import mlflow
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import mlflow.lightgbm

# Collect data into a Pandas array for training. Since the timestamp columns would likely
# cause the model to overfit the data, exclude them to avoid training on them.
columns = [col for col in training_df.columns if col not in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']]
data = training_df.toPandas()[columns]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

mlflow.lightgbm.autolog()
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
num_rounds = 100

# Train a lightGBM model
model = lgb.train(param, train_lgb_dataset, num_rounds)

# COMMAND ----------

# DBTITLE 1, Log model and return output.
# Take the first row of the training dataset as the model input example.
input_example = X_train.iloc[[0]]

# Log the trained model with MLflow
mlflow.lightgbm.log_model(
    model, 
    artifact_path="lgb_model", 
    # The signature is automatically inferred from the input example and its predicted output.
    input_example=input_example,    
    registered_model_name=model_name
)

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)

# COMMAND ----------

input_example = X_train.iloc[[0]]

mlflow.lightgbm.log_model(
    model, 
    artifact_path="lgb_model", 
    # The signature is automatically inferred from the input example and its predicted output.
    input_example=input_example,    
    registered_model_name="model_name"
)
