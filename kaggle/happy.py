# MITx 15.074x Spring 2014 Kaggle Competition
# David Wihl

import csv
import numpy as np
from sklearn import preprocessing

def preprocess(headers,array):
    # take a raw array converted from CSV, clean it up, and return as a numpy array
    le = preprocessing.LabelEncoder()

    clean = np.array([])
    userIds = []
    newRow = [0] * len(headers)
    YOB = []

    factors = []
    labelEncoders = []
    for i in xrange(len(headers)):
        factors.append(set([]))
        labelEncoders.append(preprocessing.LabelEncoder())

    target = []
    
    for row in array:
        # UserId
        userIds.append(row[0])

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
        target.append(row[7])
        
        colnum = 5
        for col in row[8:-1]: # omit last column (votes)
            factors[colnum].add(col)
            colnum += 1

    # Normalize appropriate columns
    # TODO YOB
    colnum = 0
    for f in factors:
        labelEncoders[colnum].fit(list(f))
        colnum += 1

    for le in xrange(len(labelEncoders)):
        print labelEncoders[le].classes_

    for i in xrange(len(array)):
        colnum = 0
        for col in range(2,6) + range(8,len(headers)):
            pass
            #newRow[colnum] = labelEncoders[col

    return clean

def main():
    csv.list_dialects()
    nrow = 0
    with open('train.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile, lineterminator='\r')
        headers = reader.next()
        trainData = []
        for row in reader:
            trainData.append(row)
        newArray = preprocess(headers,trainData)

    print "train has ", len(trainData), "rows"

    nrow = 0
    with open('test.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile, lineterminator='\r')
        for row in reader:
            nrow += 1
    print "test has ", nrow, "rows"



if __name__ == "__main__":
    main()
