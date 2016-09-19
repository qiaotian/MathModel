"""
te Model test case.
Author : QiaoTian
Date : 16th Sep 2016
Revised: 16th Sep 2014

""" 

import pandas as pd
import numpy as np
import csv as csv

feature_cols = []
with open('genotype.csv') as f:
    firstline = f.readline()
    feature_cols = np.array(firstline.split(','))
    assert(feature_cols.shape==(9445,))
genotype_df = pd.read_csv('genotype.csv', header=0, names=feature_cols)

assert(genotype_df.shape == (1000, 9445))
pheno_df = pd.read_csv('phenotype.txt', header=None)
assert(pheno_df.shape == (1000,1))
multi_phenos_df = pd.read_csv('multi_phenos.txt', header=None)

# train data
X = genotype_df # dataframe
y = pheno_df    # dataframe

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

"""
# default split is 75% for training and 25% for testing
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

X_train = X_train.as_matrix(feature_cols) # numpy.array
y_train = y_train.as_matrix()             # numpy.array
X_test = X_test.as_matrix(feature_cols)   # numpy.array
y_test = y_test.as_matrix()               # numpy.array



# transform X and y to numpy array
X = X.as_matrix(feature_cols)
variances = np.array([])
for idx in range(len(feature_cols)):
    print idx
    variances = np.var(X, axis=0)
print variances.shape
