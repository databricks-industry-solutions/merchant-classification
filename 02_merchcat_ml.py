# Databricks notebook source
# MAGIC %md
# MAGIC # Learning merchants
# MAGIC We will start our modelling with Occam's Razor in mind, simplicity is desired. Our first model will only use default parameters of [`fasttext`](https://fasttext.cc/) algorithm with only 5% of the available traing data as introduced in the previous notebook. This model will be our baseline model, and anything additional level of complexity should only improve the performance of our initial model. 

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous notebook, we generated a training file compatible with `fasttext` algorithm. We can load the latest file available to date alongside our validation data. Although we stored our files in a distributed location (e.g. dbfs:), storage location that must be mounted as DISK to be read as-is accross executors (more information can be found [here](https://docs.databricks.com/data/data-sources/aws/amazon-s3.html#mount-an-s3-bucket) for AWS and [here](https://docs.databricks.com/data/data-sources/azure/azure-storage.html) for Azure)

# COMMAND ----------

display(dbutils.fs.ls(config['model']['train']['hex']))

# COMMAND ----------

# distributed storage must be mounted and accessible as a file
# we ensured files were coalesced into only 1 partition so the whole training set can be read as-is
import re
training_file = dbutils.fs.ls(f"{config['model']['train']['hex']}/final")[0].path
training_file = re.sub('dbfs:', '/dbfs', training_file)

# COMMAND ----------

# MAGIC %md
# MAGIC We also load our validation set generated earlier. This will be used to evaluate our model accuracy. Given our sampling strategy, we want to know how many records to we have at our disposal to learn merchants from by joining training and testing set. 

# COMMAND ----------

validation_data = (
  spark
    .read
    .format("delta")
    .load(config['model']['train']['raw'])
    .groupBy('tr_merchant')
    .count()
    .join(spark.read.format("delta").load(config['model']['test']['raw']), ['tr_merchant'], 'left')
    .orderBy('count')
    .withColumnRenamed('count', 'training_records')
)

validation_pdf = validation_data.toPandas()
display(validation_pdf[['tr_description_clean', 'tr_merchant', 'training_records']].sample(100))

# COMMAND ----------

input_features = validation_pdf["tr_description_clean"]
input_targets  = validation_pdf["tr_merchant"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fasttext anatomy
# MAGIC Before we engineer our solution and address the challenge of serialization defined later, let's make sure we have a baseline model and understand the different moving parts. See [documentation](https://fasttext.cc/docs/en/python-module.html#train_supervised-parameters) for more information about `fasttext` and its hyper parameters. 

# COMMAND ----------

import fasttext

model = fasttext.train_supervised(
    input=training_file,
    lr=0.1,
    dim=100,
    ws=5,
    epoch=5,
    minCount=1,
    minCountLabel=1,
    minn=0,
    maxn=0,
    neg=5,
    wordNgrams=5,
    loss="softmax",
    bucket=2000000,
    thread=4,
    lrUpdateRate=100,
    t=0.0001,
    label="__label__",
    verbose=2
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can retrieve the accuracy of our model against each of its predicted classes. 

# COMMAND ----------

import re
result = input_targets.to_frame()
result.columns = ["pr_merchant"]

def predict_label(desc):
  prediction = model.predict(desc)[0][0]
  prediction = re.sub('__label__', '', prediction)
  prediction = re.sub('-', ' ', prediction)
  return prediction

# aggregate correct predictions
result["prediction"] = input_features.apply(lambda x: predict_label(x))
result["accuracy"] = result["prediction"] == result["pr_merchant"]
result["accuracy"] = result["accuracy"].apply(lambda x: float(x))
accuracies = result.groupby(["pr_merchant"])["accuracy"].mean()

# display predicted merchants
df = accuracies.to_frame().sort_values(by='accuracy', ascending=False)
df['pr_merchant'] = accuracies.index
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Given the number of merchants to learn from (1000), we will aggregate statistics for different quantiles.

# COMMAND ----------

metrics = [
    ["avg__acc", accuracies.mean()],
    ["q_05_acc", accuracies.quantile(0.05)],
    ["q_25_acc", accuracies.quantile(0.25)],
    ["q_50_acc", accuracies.median()],
    ["q_75_acc", accuracies.quantile(0.75)],
    ["q_95_acc", accuracies.quantile(0.95)]
]

import pandas as pd
display(pd.DataFrame(metrics, columns=['metric', 'value']))

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd

df = pd.DataFrame(accuracies)
df['pr_merchant'] = df.index
display(spark.createDataFrame(df[['pr_merchant', 'accuracy']]).orderBy(F.desc('accuracy')))

# COMMAND ----------

# MAGIC %md
# MAGIC Close to a 100% match for a few merchants, our model accuracy drops significantly for many brands resulting in a close to zero median score, probably explained by the diversity of characters used across merchant narratives and / or the large disparity in available data for different merchants. In the next sections, we will play with different parameters and training files of different sample sizes to ensure greater coverage of merchant with a decent accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pyfunc model
# MAGIC Before we can tweak our model any further, we want to benefit from a higher governance framework by integrating our model training with MLflow so that we can keep track of parameters and metrics. Unfortunately for us, compare to many ML toolkit, `fasttext` models are not serializable via cloudpickle format and therefore cannot be tracked via MLflow out of the box. Fortunately, mlflow comes with `pyfunc` classes that we can use to overcome this issue. Instead of serialising model as an artifact in mlflow we will use a "shell model approach". We will track parameters, metrics and a location where we have store the model in a distributed storage like `/dbfs`. Note the `clear_context` method that ensures our in-memory model is to be disposed prior to the MLFlow serialization

# COMMAND ----------

from utils.merchcat_utils import *

# COMMAND ----------

# As fasttext models cannot be automatically pickled by using cloudpickle, we will be storing the model in /dbfs
# This distributed storage was mounted to disk to be writable from any executor
fasttext_home = f"/dbfs{config['model']['path']}"

# Create model directory if it does not yet exist
dbutils.fs.mkdirs(config['model']['path'])

# COMMAND ----------

# MAGIC %md
# MAGIC We bootstrap our `fasttext` model with default hyperparameters, only specifying input data location and model output location.

# COMMAND ----------

params = {
    "input": training_file,
    "model_location": fasttext_home
}

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

with mlflow.start_run(run_name='fasttext-model') as run:
  
  # get mlflow run ID
  run_id = run.info.run_id
  
  # log parameters
  mlflow.log_params(params)
  
  # train model
  fasttextMLF = FastTextMLFlowModel(params, run_id)
  fasttextMLF.train()
  
  # evaluate model
  metrics = fasttextMLF.evaluate(input_features, input_targets)
  mlflow.log_metrics(metrics)
  
  # log model with signature
  input_schema = Schema([ColSpec("string", "input")])
  output_schema = Schema([ColSpec("string", "pr_merchant")])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  
  # dispose model prior to serialization
  fasttextMLF.clear_context()
  
  # serialize pyfunc model
  mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=fasttextMLF, 
    signature=signature
  )

# COMMAND ----------

# MAGIC %md
# MAGIC We can easily extract some metrics from our python `fasttextMLF.evaluate` function. Since we did not change any parameter, we obviously expect similar metrics as earlier but our engineering approach now allows us to track those metrics accross multiple MLFlow experiments and achieve better accuracy over time.

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
metrics = client.get_run(run_id).data.metrics
df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
df['metric'] = df.index
display(df[['metric', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC With 5% training data sample we have obtained a model with metrics presented above. From these metrics we can conclude that our model (despite a limited amount of data to learn from) was able to learn from at least 25% of merchants with a decent accuracy. Median/average averages are not a desired level and we have at least 50% of merchants we are not able to detect at all. In the next section, we will leverage [hyperopts](http://hyperopt.github.io/hyperopt/) to tune our model with different parameters so that we can verify if changing parameters affects peformance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter tuning
# MAGIC We are finally in position to talk about model performance. Although we could manually re-train models with different parameters (such as the number of `epoch` as well as `ngrams`), we could delegate that exhaustive task to `hyperopt` with minimal overhead. [Hyperopt](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt) is a framework that can perform hyper parameter tuning on top of spark. 

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, pyll
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

def train_and_log_fasttext(run, run_id, params):
  
  fasttext_params = {
      "input": params['training_file'],
      "model_location": fasttext_home,
      "lr": params['lr'],
      "epoch": int(params['epochs']),
      "wordNgrams": int(params['ngram_size']),
      "dim": int(params['dimensions'])
  }
    
  # create our model
  fasttextMLF = FastTextMLFlowModel(fasttext_params, run_id)
  
  # log parameters
  mlflow.log_params(fasttext_params)
  
  # we stored the sample size as a file name
  mlflow.log_param("sample-size", params['training_file'].split('.txt')[0].split('-')[-1])
  
  # train model
  fasttextMLF.train()
  
  # evaluate metrics
  metrics = fasttextMLF.evaluate(input_features, input_targets)
  mlflow.log_metrics(metrics)
  
  # log model with signature
  input_schema = Schema([ColSpec("string", "input")])
  output_schema = Schema([ColSpec("string", "pr_merchant")])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  
  # dispose model prior to serialization
  fasttextMLF.clear_context()
  
  # serialize pyfunc model
  mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=fasttextMLF, 
    signature=signature
  )

  # return our loss function
  loss = -metrics['avg__acc']
  return {'loss': loss, 'status': STATUS_OK, 'params': fasttext_params, 'run_id': run_id}

# COMMAND ----------

def hyper_train_model(params):
  with mlflow.start_run(run_name='fasttext-model', nested=True) as run:
    run_id = run.info.run_id
    run_result = train_and_log_fasttext(run, run_id, params)
    return run_result

# COMMAND ----------

# MAGIC %md
# MAGIC In order to train multiple models with `hyperopt` and spark, we need to define a search space and spark trials. For that purpose, we will train X models at a time (25 models in total) with a complex search space defined over 6 dimensions. Many algorithms, including `fasttext`, are able to leverage multiple threads on a given machine. Spark on the other hand assumes each task requires only a single thread to be executed. If we leave spark's default settings we will run a single model on a single node. This will considerably slow down training time for an individual run of a model. Instead, we can have a cluster dedicated for our hyper parameter tuning task. We will create a 5 node cluster where each node will have 8 cores and will set `spark.task.cpus` to 8 as well. This will let know `hyperopt` and spark run exactly one model per worker node. 

# COMMAND ----------

search_space = {
  'training_file': training_file,
  'lr': hp.uniform('lr', 0.05, 0.4),
  'epochs': hp.quniform('epochs', 5, 15, 1),
  'ngram_size': hp.quniform('ngram_size', 2, 4, 1),
  'dimensions': hp.quniform('dimensions', 20, 120, 10)
}

# COMMAND ----------

spark_trials = SparkTrials(parallelism=config['model']['executors'], spark_session=spark)

argmin = fmin(
  fn=hyper_train_model,
  space=search_space,
  algo=tpe.suggest,
  max_evals=25,
  trials=spark_trials
)

# COMMAND ----------

# MAGIC %md
# MAGIC Using `MlflowClient` combined with `spark_trials`, we can programmatically retrieve model accuracy for our best performing model. As reported below, we've achieved a perfect accuracy for at least 25% of our records with still a 90% accuracy in the worst 5% events. Overall, we've been able to successfully predict merchant names 97% of the time for a thousands of brands and million of card transactions.

# COMMAND ----------

from mlflow.tracking import MlflowClient

best_run_id = spark_trials.best_trial['result']['run_id']
client = mlflow.tracking.MlflowClient()

best_metrics = client.get_run(best_run_id).data.metrics
best_metrics.pop('loss')

df = pd.DataFrame.from_dict(best_metrics, orient='index', columns=['value'])
df['metric'] = df.index
display(df[['metric', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC We also access the best parameters using `space_eval` function of `hyperopts`, retrieving our best model based on empirical results rather than expert opinion. As represented below, the best experiment exhibiting above metrics was realized using 14 `epochs` and an `ngram_size` of 3

# COMMAND ----------

from hyperopt import space_eval
best_model_params = space_eval(search_space, argmin)
df = pd.DataFrame.from_dict(best_model_params, orient='index', columns=['value'])
df = df.astype(str)
df['param'] = df.index
display(df[['param', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we can compare all of our experiments side by side using the MLFlow user interface to better understand the effect each parameter has on our overall model accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/merchant-classification/main/images/merchcat_hyperopts_1.png" width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC Using MLFlow and `pyfunc`, we've been able to train a model that would correctly classify thousands of merchant names hidden behind million of card transaction narratives. However, this approach is based on the assumption that one already has cleaned merchant names to learn from. Although we've manually labelled tousands of card transactions with actual brand information to get started, we recognize the efforts required for such an exercise. The size and quality of labels required will be evaluated in the next section with actual empirical results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smallest training data?
# MAGIC As previously discussed, we only used 5% sample of our initial data. We may wonder if labelling more card transactions could lead to significant model improvement. Would labelling a few thousands more transaction be worth the efforts? For this purpose we will again leverage `hyperopt` and spark. We will generate many different sub samples of our training data ranging between 5% and 30% using our utility notebook introduced in the previous notebook (that we import here as a `%run` command). We will parametrize location of these files as another hyper parameter in our model optimization strategy.

# COMMAND ----------

# MAGIC %run ./utils/fasttext_utils

# COMMAND ----------

tf = TrainingFile(
    dataframe_location=config['model']['train']['raw'],
    output_location=config['model']['train']['hex'],
    target_column='tr_merchant',
    fasttext_column='fasttext'
)

file_thresholds = [0.3, 0.25, 0.2, 0.15, 0.10, 0.05]
training_files = [tf.generate_training_file(sample_rate=t, min_count=50) for t in file_thresholds]
training_files = [f'/dbfs{training_file}' for training_file in training_files]

# COMMAND ----------

# MAGIC %md
# MAGIC For this exercise we will train 90 models in parrallel. We will use `hp.choice` function that allows us to select one option from a collection of possible sample location in order to provide different training file for different experiments.

# COMMAND ----------

search_space = {
  'training_file': hp.choice('training_file', training_files),
  'lr': hp.uniform('lr', 0.05, 0.4),
  'epochs': hp.quniform('epochs', 5, 15, 1),
  'ngram_size': hp.quniform('ngram_size', 2, 4, 1),
  'dimensions': hp.quniform('dimensions', 20, 120, 10)
}

spark_trials = SparkTrials(parallelism=config['model']['executors'], spark_session=spark)
  
argmin = fmin(
  fn=hyper_train_model,
  space=search_space,
  algo=tpe.suggest,
  max_evals=90,
  trials=spark_trials
)

# COMMAND ----------

# MAGIC %md
# MAGIC Once again, we can compare experiments side by side with a new input parameter that defines our training sample size

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/merchant-classification/main/images/merchcat_hyperopts_2.png" width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC Our exploration has proven to be fruitful. We have managed to **maintain the desired predictive performance with as little as 30% of the initial training data**. If we inspect the number of rows we have in the tail merchants we actually can see that there are merchants with as low as 44 labeled rows. This implies that we could reduce our learning data even further while maintaining good performance. 

# COMMAND ----------

from mlflow.tracking import MlflowClient

best_run_id = spark_trials.best_trial['result']['run_id']
client = mlflow.tracking.MlflowClient()

best_metrics = client.get_run(best_run_id).data.metrics
best_metrics.pop('loss')

df = pd.DataFrame.from_dict(best_metrics, orient='index', columns=['value'])
df['metric'] = df.index
display(df[['metric', 'value']])

# COMMAND ----------

from hyperopt import space_eval
best_model_params = space_eval(search_space, argmin)
df = pd.DataFrame.from_dict(best_model_params, orient='index', columns=['value'])
df = df.astype(str)
df['param'] = df.index
display(df[['param', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model inference
# MAGIC Before we can infer some merchants from our original set of input transactions, let us register our best experiment as our model candidate on MLRegistry. Although our models would need to be reviewed in real life scenario, we make it available as a production artifact, programmatically. Organizations would be able to simply create web-hooks on MLFlow to notify their independant validation units (IVU process) of a new model to review prior to promoting any model to upper end environments.

# COMMAND ----------

model_uri = f'runs:/{best_run_id}/model'
result = mlflow.register_model(model_uri, config['model']['name'])
version = result.version

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=config['model']['name'],
    version=version,
    stage="Production"
)

# COMMAND ----------

logged_model = f"models:/{config['model']['name']}/production"
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load as Pandas
# MAGIC For scoring small sample that can fit in pandas dataframe we will use `model.predict` method.

# COMMAND ----------

test_pdf = validation_pdf.head(5000).sample(1000)
test_pdf["input"] = test_pdf["tr_description_clean"]
test_pdf["pr_merchant"] = loaded_model.predict(test_pdf)
display(test_pdf[["tr_description", "tr_merchant", "pr_merchant"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load as Spark
# MAGIC For scoring larger dataframes that are available via spark we will use spark `udf` that is automatically generated for us by mlflow.

# COMMAND ----------

merchant = mlflow.pyfunc.spark_udf(
  spark, 
  model_uri=logged_model, 
  result_type="string"
)

spark_results = validation_data.withColumn('pr_merchant', merchant("tr_description_clean"))
display(spark_results.select("tr_description", "tr_merchant", "pr_merchant"))

# COMMAND ----------

# MAGIC %md
# MAGIC We these two approaches we are equipped for covering micro batches as well as large historic jobs that need to process hundreds of million transactions, or more. The benefit of a spark `udf` API provided in our model wrapper class unlocks a structured streaming approach as well where card transactions can be enriched with brand information in real time. We also want to see how our model behave in real life scenario, counting the number of accurate predictions. 

# COMMAND ----------

display(
  spark_results
    .withColumn("predicted", F.when(F.col("pr_merchant") == F.col("tr_merchant"), F.lit(1)).otherwise(F.lit(0)))
    .groupBy("tr_merchant")
    .agg(F.sum(F.col("predicted")).alias("predicted"))
    .join(spark_results.groupBy("tr_merchant").count(), ["tr_merchant"])
    .withColumn("accuracy", F.col("predicted") / F.col("count"))
    .orderBy(F.desc("accuracy"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC As represented above, our model coverage significantly improved since our first iteration. The balance of coverage / accuracy would need to be constantly monitored as new card transactions unfold. Ideally, through the framework defined here, organizations could automatically (or with minimum supervision) learn from new patterns and new labels (as set by e.g. end users through their mobile banking applications) once the quality of output is no longer met.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC In this solution we have focused on merchant classification problem approching it from short document classification angle. For this task we have selected [`fasttext`](https://fasttext.cc/) as model of choice, we have successfuly integrated this model with MLFlow and `hyperopt`. As part of this exercise, we have demonstrated that an organization can start **introducing good quality merchant classification with as little as 50 labeled record per merchant** (five zero only!). This fact unlock a lot of value. 

# COMMAND ----------

# MAGIC %md
# MAGIC A team of analysts can spend as few as several days labeling the initial source of truth before this automated solution can take over. The analysts can then switch into "autopilot" mode and start focusing on added value that we can extract from transactional data such as fraud or customer spending patterns. With a robust transaction enrichment in place, we can present transactions in our mobile banking with merchants properly identified and with confidence that what we presenting is correct (hence maintaining high quality and trust with our end customers) as per above example. In a next solution accelerator, we will be using this classification as a building block to drive personalized insights and behavioral transactions patterns (transaction embeddings).
