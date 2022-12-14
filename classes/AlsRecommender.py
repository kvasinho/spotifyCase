import warnings
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
import numpy as np


class ALSRecommender:
    def __init__(self):
        self.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("SparkByExamples.com") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")

        self.trained = 0

    def __str__(self):
        return f"Pyspark ALS Algorithm. Train on a new dataset or recommend new Items"

    def splitTrainTest(self,
                       table,
                       testSize=0.3,
                       verbose=True):

        self.table = table

        (self.train, self.test) = self.table.randomSplit([1 - testSize, testSize])
        if verbose:
            return f"Train and Test sets with a Test Size of {testSize} created."

    def trainModel(self,
                   noFeatures=15,
                   iterations=5,
                   implicit=True,
                   reg=0.1, a=0.1,
                   user="uid",
                   item="tid",
                   rating="rating",
                   nonneg=True,
                   verbose=True):

        als = ALS(maxIter=iterations,
                  rank=noFeatures,
                  implicitPrefs=implicit,
                  regParam=reg,
                  alpha=a,
                  userCol=user,
                  itemCol=item,
                  ratingCol=rating,
                  coldStartStrategy="drop",
                  nonnegative=nonneg)

        self.model = als.fit(self.train)

        items = self.model.itemFactors.orderBy("id").select("features").collect()
        items = [item["features"] for item in items]
        self.itemFeatures = np.array(items)

        users = self.model.userFactors.orderBy("id").select("features").collect()
        users = [item["features"] for item in users]
        self.userFeatures = np.array(users)

        self.trained = 1

        if verbose:
            return f"ALS model trained. ItemFeatures: {self.itemFeatures.shape}"

    def evaluate(self,
                 metric="rmse"):
        self.predictions = self.model.transform(self.test)
        evaluator = RegressionEvaluator(metricName=metric, labelCol="rating",
                                        predictionCol="prediction")
        self.score = evaluator.evaluate(self.predictions)
        return f"{metric} on the test set: {self.score}"

    def recommendForTrainedUsers(self,
                                 itemFeatures=None,
                                 n=10):
        # if itemFeatures==None:
        assert self.trained == 1, "No array with features specified. Insert a Numpy Array with features or Train the model."
        return self.model.recommendForAllUsers(n).show()

    # else:

    def recommendForTrainedSubset(self,
                                  itemFeatures=None,
                                  n: int = 10,
                                  subset: list = [0]):
        # if itemFeatures==None:
        assert self.trained == 1, "No array with features specified. Insert a Numpy Array with features or Train the model."
        return self.model.recommendForUserSubset(subset, n).show()

    # else:

    def recommendForNewUser(self,
                            items=[0, 1, 2],
                            matrix=None,
                            n=10,
                            convertToNames=False,
                            indexDict=None):
        if isinstance(matrix, np.ndarray):
            f = matrix
        else:
            f = self.itemFeatures

        relevantItems = items

        solverVector = [f[item] for item in relevantItems]
        ratingsVector = np.ones(len(items))

        newUserFeatures = np.linalg.lstsq(solverVector,
                                          ratingsVector,
                                          rcond=None)[0]  # have to use this because this equation has no exact solution (besides the case of len(items)==model.rank)

        predictions = np.dot(newUserFeatures, np.transpose(f))

        indexPredictionDict = dict(zip(list(range(len(predictions))), list(predictions)))
        predictionIndexDict = dict(zip(list(predictions), list(range(len(predictions)))))

        deleted = [indexPredictionDict.pop(key) for key in items]

        topN = sorted(list(indexPredictionDict.values()), reverse=True)[0:n]
        topIndices = [predictionIndexDict[item] for item in topN]

        if convertToNames:
            assert isinstance(indexDict, dict), "No dictionary with indices specified. Please specify a dict with the format 'index':'name'"
            return [indexDict[item] for item in topIndices]
        else:
            return topIndices

