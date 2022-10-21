# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/merchant-classification/tree/web-sync. For more information about this solution accelerator, visit [this blog](https://www.databricks.com/blog/2021/05/10/improving-customer-experience-with-transaction-enrichment.html).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![DBU](https://img.shields.io/badge/DBU-XL-red)]()
# MAGIC [![COMPLEXITY](https://img.shields.io/badge/COMPLEXITY-301-red)]()
# MAGIC 
# MAGIC 
# MAGIC *According to 2020 research from the [Nilson Report](https://nilsonreport.com/), roughly 1 billion card transactions occur every day around the world (100 million transactions in the US alone). That is 1 billion data points that can be exploited every day by retail banks and payment processor companies to better understand customers spending behaviours, improve customer experience through their mobile banking applications, drive significant cross sell opportunities in the context of customer 360 or leverage these personalized insights to reduce fraud. With many entities involved in the authorization and settlement of a card transaction, the contextual information carried forward from a merchant to a retail bank is complicated, sometimes misleading and requires the use of advanced analytics techniques to extract clear brand and merchant information. In this solution accelerator, we demonstrate how the lakehouse architecture enables banks, open banking aggregators and payment processors to address the core challenge of retail banking: merchant classification.*
# MAGIC 
# MAGIC 
# MAGIC ___
# MAGIC <milos.colic@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/merchcat/images/reference_architecture.png" width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                               | description             | license    | source                                              |
# MAGIC |-------------------------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | fasttext                                              | NLP library             | BSD License| https://fasttext.cc/                                   |
