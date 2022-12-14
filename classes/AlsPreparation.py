import warnings
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

class ALSPreparator:
    warnings.filterwarnings("ignore")

    def __init__(self):
        self.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("SparkByExamples.com") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")

    def __str__(self):
        return f"Converts the data to a format which is ready to be used by the ALSRecommender"

    def readFromCsvConverted(self, file, head=True, user="uid", item="tid", rating="rating"):
        schema = StructType([
            StructField(user, IntegerType(), True),
            StructField(item, IntegerType(), True),
            StructField(rating, IntegerType(), True),
        ])
        self.ratings = self.spark.read.csv(file, header=head, schema=schema)

    def readFromCSVRaw(self, file, head=True, user="user", item="item", rating="rating", itemDicts=True, userDicts=True):
        schema = StructType([
            StructField(user, StringType(), True),
            StructField(item, StringType(), True),
            StructField(rating, IntegerType(), True),
        ])

        df = self.spark.read.csv(file, header=head, schema=schema)

        uIndexer = StringIndexer(inputCol="user", outputCol="uid")
        iIndexer = StringIndexer(inputCol="item", outputCol="tid")

        # Fit the StringIndexer and transform the DataFrame
        indexed = uIndexer.fit(df).transform(iIndexer.fit(df).transform(df))

        indexed = indexed.withColumn("tid", indexed["tid"].cast("integer"))
        indexed = indexed.withColumn("uid", indexed["uid"].cast("integer"))

        if userDicts:
            self.indexUserDict = indexed.select("uid", "user").rdd.map(lambda x: (x[0], x[1])).collectAsMap()
            self.userIndexDict = indexed.select("user", "uid").rdd.map(lambda x: (x[0], x[1])).collectAsMap()
        if itemDicts:
            self.indexItemDict = indexed.select("tid", "item").rdd.map(lambda x: (x[0], x[1])).collectAsMap()
            self.itemIndexDict = indexed.select("item", "tid").rdd.map(lambda x: (x[0], x[1])).collectAsMap()

        self.dataFrame = indexed.select(["uid", "tid", "rating"])

        return "table converted"


