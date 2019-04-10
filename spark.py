import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# userDF = pd.read_csv("data_clean.csv")

spark = SparkSession\
    .builder\
    .appName("Football")\
    .getOrCreate()

userDF = spark\
    .read\
    .format("csv")\
    .options(header="true", inferSchema="true")\
    .load("data_clean.csv")

print(userDF.head())