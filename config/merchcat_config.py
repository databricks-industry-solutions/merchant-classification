# Databricks notebook source
# MAGIC %pip install fasttext==0.9.2

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import re
from pathlib import Path

# We ensure that all objects created in that notebooks will be registered in a user specific database.
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username = useremail.split('@')[0]

# Please replace this cell should you want to store data somewhere else.
database_name = '{}_merchcat'.format(re.sub('\W', '_', username))
_ = sql("CREATE DATABASE IF NOT EXISTS {}".format(database_name))

# Similar to database, we will store actual content on a given path
home_directory = '/FileStore/{}/merchcat'.format(username)
dbutils.fs.mkdirs(home_directory)

# Where we might stored temporary data on local disk
temp_directory = "/tmp/{}/merchcat".format(username)
Path(temp_directory).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

import re

config = {
  'num_executors'             :  '8',
  'model_name'                :  'merchcat_{}'.format(re.sub('\.', '_', username)),
  'transactions_raw'          :  '/mnt/industry-gtm/fsi/datasets/card_transactions',
  'transactions_fasttext'     :  '{}/labeled_transactions'.format(home_directory),
  'transactions_model_dir'    :  '{}/fasttext'.format(home_directory),
  'transactions_train_raw'    :  '{}/labeled_transactions_train_raw'.format(home_directory),
  'transactions_train_hex'    :  '{}/labeled_transactions_train_hex'.format(home_directory),
  'transactions_valid_raw'    :  '{}/labeled_transactions_valid_raw'.format(home_directory),
  'transactions_valid_hex'    :  '{}/labeled_transactions_valid_hex'.format(home_directory)
}

# COMMAND ----------

import pandas as pd
 
# as-is, we simply retrieve dictionary key, but the reason we create a function
# is that user would be able to replace dictionary to application property file
# without impacting notebook code
def getParam(s):
  return config[s]
 
# passing configuration to scala
spark.createDataFrame(pd.DataFrame(config, index=[0])).createOrReplaceTempView('esg_config')

# COMMAND ----------

import mlflow
experiment_name = f"/Users/{useremail}/merchcat_experiment"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %scala
# MAGIC val cdf = spark.read.table("esg_config")
# MAGIC val row = cdf.head()
# MAGIC val config = cdf.schema.map(f => (f.name, row.getAs[String](f.name))).toMap
# MAGIC def getParam(s: String) = config(s)

# COMMAND ----------

def tear_down():
  import shutil
  shutil.rmtree(temp_directory)
  dbutils.fs.rm(home_directory, True)
  _ = sql("DROP DATABASE IF EXISTS {} CASCADE".format(database_name))
