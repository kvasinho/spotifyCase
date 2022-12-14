from classes.AlsRecommender import ALSRecommender
from classes.AlsPreparation import ALSPreparator


def main():
    a = ALSPreparator()
    a.readFromCSVRaw("testCsv.csv")
    df = a.dataFrame

    r = ALSRecommender()
    r.splitTrainTest(testSize=0.2, table=a.dataFrame)
    r.trainModel()
    print(r.recommendForNewUser([5, 0, 1, 7], n=5, convertToNames=True, indexDict=a.indexItemDict))
    print(r.recommendForNewUser([5, 0, 1, 7], n=5, convertToNames=False))
    print(a.indexItemDict)

if __name__=="__main__":
    main()
