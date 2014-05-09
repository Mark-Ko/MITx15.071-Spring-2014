# MITx 15.074x Spring 2014 Kaggle Competition
# David Wihl

import csv
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn import metrics

global labelEncoders

class Dataset(object):
    def __init__(self):
        self.labelEncoders = []

    def load(self):
        self.labelEncoders = pickle.load( open("encoders.p", "rb") )

    def save(self):
        pickle.dump(self.labelEncoders, open("encoders.p", "wb") )

    def trainProcess(self,headers, array):
        newRow = [0] * len(headers)
        YOB = []

        factors = []

        for i in xrange(len(headers)):
            factors.append(set([]))
            self.labelEncoders.append(preprocessing.LabelEncoder())

        target = []
    
        for row in array:
            # YOB - to be normalized
            if row[1] != 'NA':
                YOB.append(row[1])

            # Gender, Income, HouseholdStatus, EducationLevel, Party
            factors[0].add(row[2])
            factors[1].add(row[3])
            factors[2].add(row[4])
            factors[3].add(row[5])
            factors[4].add(row[6])

            # Target (Happy = {0,1})
            target.append(int(row[7]))
                
            colnum = 5
            for col in row[8:-1]: # omit last column (votes)
                factors[colnum].add(col)
                colnum += 1

        # Encode Labels
        colnum = 0
        for f in factors:
            self.labelEncoders[colnum].fit(list(f))
            colnum += 1

        # ok, let's add a new cleaned up row 
        rownum = 0
        cleanArray = []
        for row in array:
            newRow = []
            colnum = 0
            if rownum % 1000 == 0 and rownum != 0:
                print rownum
            #demographics
            for col in row[2:7]:
                newRow.append(self.labelEncoders[colnum].transform([col])[0])
                colnum += 1

            # 101 questions
            for col in row[8:-1]:
                newRow.append(self.labelEncoders[colnum].transform([col])[0])
                colnum += 1

            cleanArray.append(newRow)
            rownum += 1

        return cleanArray, target


    def testProcess(self,headers, array):
        userIds = []

        YOB = []

        # ok, let's add a new cleaned up row 
        rownum = 0
        cleanArray = []
        for row in array:
            userIds.append(row[0])

            newRow = []
            colnum = 0
            if rownum % 1000 == 0 and rownum != 0:
                print rownum

            for col in row[2:-1]:
                newRow.append(self.labelEncoders[colnum].transform([col])[0])
                colnum += 1

            cleanArray.append(newRow)
            rownum += 1

        return userIds, cleanArray

def main():
    nrow = 0
    ds = Dataset()
    print "reading and processing..."
    try:
        X = np.load("X.npy")
        y = np.load("y.npy")
        ds.load()
    except:
        print "exception thrown"
        with open('train.csv', 'rU') as csvfile:
            reader = csv.reader(csvfile, lineterminator='\r')
            headers = reader.next()
            trainData = []
            for row in reader:
                trainData.append(row)
            X_array, target = ds.trainProcess(headers,trainData)


        X = np.array(X_array)
        y = np.array(target)
        np.save("X.npy",X)
        np.save("y.npy",y)
        ds.save()

    print "Starting train..."
    print "X shape", X.shape

    clf = RandomForestRegressor(n_estimators=2000)
    scores = cross_val_score(clf, X, y, cv = 5)
    print scores.mean()
    return 

    clf.fit(X,y)

    print "Starting predictions..."

    nrow = 0
    with open('test.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile, lineterminator='\r')
        headers = reader.next()
        testData = []
        for row in reader:
            testData.append(row)
        userIds, testArray = ds.testProcess(headers, testData)

    X_test = np.array(testArray)
    print "X test shape", X_test.shape

    pred = clf.predict(X_test)

    with open('subextra.csv','wb') as f:
        writer = csv.writer(f)
        writer.writerow(['UserID','Probability1'])
        for u, p in zip(userIds, pred):
            if p > 0.99: p = 0.99
            if p < 0.001: p = 0.001
            writer.writerow([u,p])

    print "Submission CSV written."


if __name__ == "__main__":
    main()

# todo:
# 1. add normalization code for YOB and questions answered, then normalize the values
# 2. Try a two or four cluster then run RF against it.
# 3. Clean out bad training data
