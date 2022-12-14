import warnings
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
import numpy as np


#this is the fun part. This can be used on every spark dataframe with the correct format.
#We need 3 columns - User,Item, Rating - works with string types as well, but takes more Ram. That's why int casting is useful
# Currently, we need train test split to read in the Dataframe. this Could be improved by reading in  the df separately
# e.g. if we wanted to train on full datasets, etc. Test Size could probably be set to 0 as well to achieve the same.

#returns either recommendations for users in the training set (test set could be included as well) or for new Users inserting relevant songs to a list
#

#Do save A LOT of time, it's possible to call the recommender on an existing Feature Matrix as well. WE´re not relying on Pyspark to compute the recommendations for new users.


#to-do:
#Leave one Out instead of train test split
#separate read-in and split
#add Error metric for new users


class ALSRecommender:
    #Pyspark initialization
    def __init__(self):
        self.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("SparkByExamples.com") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")

        self.trained = 0

    def __str__(self):
        return f"Pyspark ALS Algorithm. Train on a new dataset or recommend new Items"

    #Read-in and split of the Dataframe, returns both splits.
    def splitTrainTest(self,
                       table,
                       testSize=0.3,
                       verbose=True):
        self.table = table
        (self.train, self.test) = self.table.randomSplit([1 - testSize, testSize])
        if verbose:
            return f"Train and Test sets with a Test Size of {testSize} created."

    #takes the parameters that I perceived to be important, By default, the columns are called uid, tid & rating. This can be adjusted if necessary.
    #However, it probably wont work when we don't specify different column names.

    #Implicit may be changed, however it's appropriate for this specific tasks with implicit ratings.
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

        #returns Item Factor Matrix to give recommendations to new users.
        items = self.model.itemFactors.orderBy("id").select("features").collect()
        items = [item["features"] for item in items]
        self.itemFeatures = np.array(items)

        #same for the User Factor Matrix. Quite useful to calculate user simmilarities. Not implemented yet.
        users = self.model.userFactors.orderBy("id").select("features").collect()
        users = [item["features"] for item in users]
        self.userFeatures = np.array(users)


        self.trained = 1

        if verbose:
            return f"ALS model trained. ItemFeatures: {self.itemFeatures.shape}"

        #Returns the Evaluation Metrics for this model. It works, but it doesnt do a lot currently.
    def evaluate(self,
                 metric="rmse"):
        self.predictions = self.model.transform(self.test)
        evaluator = RegressionEvaluator(metricName=metric, labelCol="rating",
                                        predictionCol="prediction")
        self.score = evaluator.evaluate(self.predictions)
        return f"{metric} on the test set: {self.score}"

        #Gives the top n recommendations for all users.
        #only works when the model was trained before
    def recommendForTrainedUsers(self,
                                 itemFeatures=None,
                                 n=10):
        # if itemFeatures==None:
        assert self.trained == 1, "No array with features specified. Insert a Numpy Array with features or Train the model."
        return self.model.recommendForAllUsers(n).show()

        # gives the top n recommendations for a specific subset of users
        # only works when the model was trained
    def recommendForTrainedSubset(self,
                                  itemFeatures=None,
                                  n: int = 10,
                                  subset: list = [0]):
        assert self.trained == 1, "No array with features specified. Insert a Numpy Array with features or Train the model."
        return self.model.recommendForUserSubset(subset, n).show()

        #USes the item feature Matrix to calculate new user features.
        # Unfortunately we almost always have some degrees of freedom, which is why there is no exact solution. We try to minimize the error
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

        #this either returns the index (to be checked manually, or the names from the index-value dict. Needs to be specified on-call
        if convertToNames:
            assert isinstance(indexDict, dict), "No dictionary with indices specified. Please specify a dict with the format 'index':'name'"
            return [indexDict[item] for item in topIndices]
        else:
            return topIndices

