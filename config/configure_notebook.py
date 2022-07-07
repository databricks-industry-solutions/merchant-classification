# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import yaml
with open('config/application.yaml', 'r') as f:
  config = yaml.safe_load(f)

# COMMAND ----------

import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/{}'.format(username, config['model']['name']))
