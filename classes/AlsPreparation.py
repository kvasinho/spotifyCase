import warnings
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

#Currently, only CSV import is supported.
# Either as raw file with 3 Columns: User (but works for playlists as well), Item, Rating
# Or as a converted file -> takes numeric (indexed columns as input - need to store a dict before, but it`s faster)
# Raw Input takes longer to compute, however it returns important dictionaries as well.

#To-Do:
# Check if the raw version works as well, when columns are numeric types. -> Maybe Pyspark converts it?
#


class ALSPreparator:
    #warnings.filterwarnings("ignore")

    #create the spark session.
    def __init__(self):
        self.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("SparkByExamples.com") \
            .getOrCreate()
        #to prevent perma warnings when the Schema and Column name doesn't match.
        self.spark.sparkContext.setLogLevel("ERROR")

    def __str__(self):
        return f"Converts the data to a format which is ready to be used by the ALSRecommender"

    ###  takes 3 numeric columns - User (-index), Item (-index), Rating - as input
    ###  returns the dataframe which can be used for pyspark ALS without the dictionaries
    def readFromCsvConverted(self, file, head=True, user="uid", item="tid", rating="rating"):
        schema = StructType([
            StructField(user, IntegerType(), True),
            StructField(item, IntegerType(), True),
            StructField(rating, IntegerType(), True),
        ])
        self.dataFrame = self.spark.read.csv(file, header=head, schema=schema)

    ### the slow version. Takes 3 Columns - User(String), Item(String), Rating(Numeric) as input
    ### returns the dataframe and dictionaries for index-value conversion
    def readFromCSVRaw(self, file, head=True, user="user", item="item", rating="rating", itemDicts=True, userDicts=True):
        schema = StructType([
            StructField(user, StringType(), True),
            StructField(item, StringType(), True),
            StructField(rating, IntegerType(), True),
        ])

        df = self.spark.read.csv(file, header=head, schema=schema)

        #StringIndexer creates an Index and converts the string column. Without Int casting, it gets converted to float.
        uIndexer = StringIndexer(inputCol="user", outputCol="uid")
        iIndexer = StringIndexer(inputCol="item", outputCol="tid")

        # Fit the StringIndexer and transform the DataFrame
        indexed = uIndexer.fit(df).transform(iIndexer.fit(df).transform(df))

        #Float to int
        indexed = indexed.withColumn("tid", indexed["tid"].cast("integer"))
        indexed = indexed.withColumn("uid", indexed["uid"].cast("integer"))

        #if specified, we return these dictionaries as well. ItemDicts should be returned pretty much every time.
        if userDicts:
            self.indexUserDict = indexed.select("uid", "user").rdd.map(lambda x: (x[0], x[1])).collectAsMap()
            self.userIndexDict = indexed.select("user", "uid").rdd.map(lambda x: (x[0], x[1])).collectAsMap()
        if itemDicts:
            self.indexItemDict = indexed.select("tid", "item").rdd.map(lambda x: (x[0], x[1])).collectAsMap()
            self.itemIndexDict = indexed.select("item", "tid").rdd.map(lambda x: (x[0], x[1])).collectAsMap()

        #drop everything else
        self.dataFrame = indexed.select(["uid", "tid", "rating"])

        return "table converted"


