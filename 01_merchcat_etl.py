# Databricks notebook source
# MAGIC %md
# MAGIC # Preprocessing
# MAGIC 
# MAGIC The transaction narrative and merchant description is a free form text filled in by a merchant without common guidelines or industry standards, hence requiring a data science approach to this data inconsistency problem. In this solution accelerator, we demonstrate how text classification techniques can help organizations better understand the brand hidden in any transaction narrative given a reference data set of merchants. How close is the transaction description `STARBUCKS LONDON 1233-242-43 2021` to the company "Starbucks"? 

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC It doesn't come as a surprise that the quality of the system outputs is directly correlated to the quality of data we are able to supply to our machine learning model. Securing a properly structured and good quality training and test data samples is as important as training a good machine learning model. We will commence our journey with a sample of raw transactional data with a merchant narrative as it might appear at a point of sale. The most basic format one can expect POS would generate is (date, amount, description, card number). 

# COMMAND ----------

from pyspark.sql import functions as F

tr_df = (
    spark
        .read
        .format('delta')
        .load(config['transactions']['raw'])
        .select('tr_date', 'tr_merchant', 'tr_description', 'tr_amount')
        .filter(F.expr('tr_merchant IS NOT NULL'))
)

display(tr_df.select("tr_date", "tr_description", "tr_amount"))

# COMMAND ----------

# MAGIC %md
# MAGIC In this solution accelerator, we want to demonstrate the use of [`fasttext`](https://fasttext.cc/) library, an efficient framework for text classification and representation learning. The aim of this notebook is to translate raw card transaction narrative into input data that can be fed into a `fasttext` model. For this exercise, we previously labelled thousands of card transactions with actual brand and merchant names (our `tr_merchant` column) that we further refined through that series of notebooks in order to create a dataset of millions of labelled transactions. In real life, most of Financial services organizations will already have a existing series of labels to learn merchants from. The size and quality of labels required will be evaluated in the next notebook with actual empirical results.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchant narrative
# MAGIC The first thing we notice is that card transaction narrative are highly unstructured. These descriptions do not follow a global format and will often contain partially purged data. Oftentimes the data will contain dates, amounts, unique identifiers and similar tokens that do not bring any valuable information when it comes to understanding merchant associated to a card transaction. Whith this in mind we have performed data cleansing activities as part of a pre-processor step. Based on kaggle [article](https://www.kaggle.com/edrushton/removing-dates-data-cleaning) for date removal from string data, we have produced a series of simple regular expressions to clean our descriptions from dates and unwanted characters we know do not carry any descriptive value.

# COMMAND ----------

from utils.regex_utils import *

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

@udf(returnType=T.StringType())
def dates_udf(description):
    return str(date_pattern.sub(" ", str(description)))

tr_df_cleaned = (
    tr_df
        .withColumn("tr_description_clean", dates_udf(F.col("tr_description")))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), price_regex, ""))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "(\(+)|(\)+)", ""))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "&", " and "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "[^a-zA-Z0-9]+", " "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "\\s+", " "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "\\s+x{2,}\\s+", " ")) 
        .withColumn("tr_description_clean", F.trim(F.col("tr_description_clean")))
)

display(tr_df_cleaned.select("tr_merchant", "tr_description", "tr_description_clean"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### fasttext format
# MAGIC As part of that cleaning and sampling exercise, we will also format our data to comply with a `fasttext` model. Fasttext model requires data in a specific format. The label to learn from is the actual merchant (`tr_merchant`) whilst the pattern is our cleansed description (`tr_description_clean`). 
# MAGIC 
# MAGIC ```
# MAGIC __label__merchant1 clean description from narrative 1
# MAGIC __label__merchant2 clean description from narrative 2
# MAGIC __label__merchant3 clean description from narrative 3
# MAGIC ```

# COMMAND ----------

tr_df_fasttext = tr_df_cleaned.withColumn(
    "fasttext",
    F.concat(
        F.concat(
            F.lit("__label__"),
            F.regexp_replace(F.col("tr_merchant"), "\\s+", "-")
        ),
        F.lit(" "),
        F.col("tr_description_clean")
    )
)

display(tr_df_fasttext.select("fasttext"))

# COMMAND ----------

# MAGIC %md
# MAGIC We will store this input dataset as a delta table that can be used for machine learning purpose.

# COMMAND ----------

_ = (
    tr_df_fasttext
      .write
      .mode("overwrite")
      .format("delta")
      .save(config['transactions']['fmt'])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imbalanced dataset
# MAGIC When it comes to card transactions data, it is very common to come across a large disparity in available data for different merchants. For example it is to be expected that "Amazon" will drive much more transactions than "MyLittleCornerShop". Let's inspect the distribution of our raw data.

# COMMAND ----------

tr_df = spark.read.format('delta').load(config['transactions']['fmt'])
df = tr_df.groupBy("tr_merchant").count().orderBy("count").toPandas()
df.plot.hist(bins=100)

# COMMAND ----------

# MAGIC %md
# MAGIC We can conclude that the data available for machine learning is very different if we are comparing "Tesco" and "MyLittleCornerShop". For Tesco we have millions of card transactions to learn from compare to a mere thousands of data points for others. So what is the right amount of data per merchant for us be able to learn text patterns from in order to score newly arriving transactions with success? The only proper way to answer this question is to pair it with metrics. We need to be able to measure our performance against each merchant and against whole population. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sampling strategy
# MAGIC If we are providing data samples that are orders of magnitude different in size our model might learn well only from larger samples and merchants with less data might end up being treated as misclassification. In order to address this issue we will use **stratification**. We will sample all our merchants to have at least 100 labeled points and up to 5000 points at maximum. 

# COMMAND ----------

def format_dict(label_column, value_column, in_dict):
    labels = in_dict[label_column]
    rates = in_dict[value_column]
    result = dict()
    for i in range(0, len(labels)):
        result[labels[i]] = rates[i]
    return result

def sample_data(sample_size, count_threshold, data):
    counted = data.groupBy("tr_merchant").count()
    counted = counted.where(F.col("count") >= count_threshold)
    counted = counted \
        .withColumn("sample_rate", sample_size / F.col("count")) \
        .withColumn("sample_rate", F.when(F.col("sample_rate") > 1, 1).otherwise(F.col("sample_rate")))
    sample_rates = counted.select("tr_merchant", "sample_rate").toPandas().to_dict()
    sample_rates = format_dict("tr_merchant", "sample_rate", sample_rates)
    result = data.sampleBy("tr_merchant", sample_rates)
    return result

# COMMAND ----------

tr_df_sampled = sample_data(5000, 100, tr_df)
df_sampled = tr_df_sampled.groupBy("tr_merchant").count().orderBy("count").toPandas()
df_sampled.plot.hist(bins=100)

# COMMAND ----------

# MAGIC %md
# MAGIC We do notice that there are brands that are under represented with respect to average available data. However we will leave them in and allow the model to try and learn from these as one of our key objectives is to determine the minimum required data to be labeled for a brand to be recognized by the model. The motivation for this is rooted in our desire to demonstrate to financial services that do not have labeled data in place **the minimal effort required to form a source of truth needed to train a machine learning model** (as we did ourselves to create that initial set of data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training set
# MAGIC Before diving into the world of ML yet, we also require to split our data into train/validation samples. One way of achieving this is attaching a per class percentile to each row in the dataset with a random ordering. This will ensure that we can extract 10% of data as a validation data in reproducible manner.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

w =  Window.partitionBy("tr_merchant").orderBy(F.rand())
df = tr_df_sampled.withColumn("class_percentile", F.bround(F.percent_rank().over(w), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC We split our data and store both training and validation set to Delta Lake tables.

# COMMAND ----------

df.where("class_percentile < 0.9") \
  .write \
  .mode("overwrite") \
  .format("delta") \
  .save(config['model']['train']['raw'])

# COMMAND ----------

df.where("class_percentile >= 0.9") \
  .write \
  .mode("overwrite") \
  .format("delta") \
  .save(config['model']['test']['raw'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## fasttext files
# MAGIC In addition to a specific format, `fasttext` model also expects to load data from a single text file. Each line in this file must be in the format we have enforced earlier. `TrainingFile` class in our util notebook manages the logic needed to convert a spark dataframe into a single flat file expected by the `fasttext` training logic. We generate our files with a unique name that we store at a specified output location readable across all executors (i.e. mounted to disk). 

# COMMAND ----------

# MAGIC %run ./utils/fasttext_utils

# COMMAND ----------

tf = TrainingFile(
    dataframe_location=config['model']['train']['raw'],
    output_location=config['model']['train']['hex'],
    target_column='tr_merchant',
    fasttext_column='fasttext'
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can generate our training file as follows. Note that we may have to generate samples of different size depending on the outcome of our initial model in our next notebook. Each generated training file with different sample size will be stored with a specific version (a UUID) that can be tracked across MLFlow experiments.

# COMMAND ----------

training_file = tf.generate_training_file(
    sample_rate=0.05, 
    min_count=50
)

# COMMAND ----------

display(dbutils.fs.ls(config['model']['train']['hex']))

# COMMAND ----------

input_dir = '{}/final'.format(config['model']['train']['hex'])
display(spark.read.format('text').load(input_dir))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Take away
# MAGIC In these first sections, we have dedicated a substantial amount of effort in cleaning and standardising our data. The motivation is simple, higher quality data will yield higher quality machine learning. With our `fasttext` training files in place, we can now train our initial model to extract merchant from card transaction narrative
