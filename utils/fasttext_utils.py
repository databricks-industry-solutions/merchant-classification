# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import uuid
import pandas as pd


class TrainingFile():

    def __init__(self, dataframe_location, output_location, target_column, fasttext_column):
        self.dataframe_location = dataframe_location
        self.output_location = output_location
        self.target_column = target_column
        self.fasttext_column = fasttext_column
        self.spark = SparkSession.getActiveSession()

    def __get_part_name(self, file_path):
        files = dbutils.fs.ls(file_path)
        files = pd.DataFrame(files)
        return files[files.name.apply(lambda n: "part" in n)]["name"].iloc[0]

    def __format_dict(self, label_column, value_column, in_dict):
        labels = in_dict[label_column]
        rates = in_dict[value_column]
        result = dict()
        for i in range(0, len(labels)):
            result[labels[i]] = rates[i]
        return result

    def generate_fixed_training_file(self, count):
        data = self.spark.read.format("delta").load(self.dataframe_location)
        w = Window.partitionBy(self.target_column).orderBy(F.rand())
        data = data.withColumn("rank", F.row_number().over(w))
        df = data.where(F.col("rank") <= count).drop("rank")
        unique_name = uuid.uuid4().hex
        file_path = f"{self.output_location}/{unique_name}"
        result_path = f"{self.output_location}/final/{unique_name}-n-{count}.txt"
        df.select(self.fasttext_column).coalesce(1).write.mode("overwrite").text(file_path)
        part_name = self.__get_part_name(file_path)
        dbutils.fs.cp(f"{file_path}/{part_name}", result_path)
        return result_path  

    def generate_training_file(self, sample_rate, min_count):
        data = self.spark.read.format("delta").load(self.dataframe_location)
        counted = data.groupBy(self.target_column).count()
        counted = counted.withColumn(
          "sample_rate", 
          F.when(
            F.col("count")*sample_rate < min_count, min_count/F.col("count") + 0.05 
            # it is better if we oversample a bit near the threshold than if we undersample
          ).otherwise(sample_rate)
        )
        sample_rates = counted.select(self.target_column, "sample_rate").toPandas().to_dict()
        sample_rates = self.__format_dict(self.target_column, "sample_rate", sample_rates)
        df = data.sampleBy(self.target_column, sample_rates)
        unique_name = uuid.uuid4().hex
        file_path = f"{self.output_location}/{unique_name}"
        t = int(100*sample_rate)
        result_path = f"{self.output_location}/final/{unique_name}-{t}.txt"
        df.select(self.fasttext_column).coalesce(1).write.mode("overwrite").text(file_path)
        part_name = self.__get_part_name(file_path)
        dbutils.fs.cp(f"{file_path}/{part_name}", result_path)
        return result_path 
