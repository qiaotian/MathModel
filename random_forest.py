""" Writing my first Graduate Model test case.
Author : QiaoTian
Date : 16th Sep 2016
Revised: 16th Sep 2014

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from minepy import MINE
from scipy.stats import pearsonr
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

feature_cols = []
with open('genotype.csv') as f:
    firstline = f.readline()
    feature_cols = np.array(firstline.split(','))

genotype_df         = pd.read_csv('genotype.csv', header=0, names=feature_cols)
pheno_df            = pd.read_csv('phenotype.txt', header=None)
multi_phenos_df     = pd.read_csv('multi_phenos.txt', header=None)

# train data
X = genotype_df
y = pheno_df

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

X_train = X_train.as_matrix(feature_cols)
y_train = y_train.as_matrix()
X_test  = X_test.as_matrix(feature_cols)
y_test  = y_test.as_matrix()


""" correlation coefficient """
K = 40
X_train_new = SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=K).fit_transform(X_train, y_train)
X_test_new = SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=K).fit_transform(X_test, y_test)

""" mutual information """
#def mic(x, y):
#    m = MINE()
#    m.compute_score(x, y)
#    return (m.mic(), 0.5)
#X_train_new = SelectKBest(lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T, k=50).fit_transform(X_train, y_train)
#X_test_new = SelectKBest(lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T, k=50).fit_transform(X_test, y_test)

""" chi-square validation """
#X_train_new = SelectKBest(chi2, k=50).fit_transform(X_train, y_train)
#X_test_new = SelectKBest(chi2, k=50).fit_transform(X_test, y_test)

""" random foreset """

error_in = []
error_out = []
n_est = 200
for it in range(200):
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                            max_depth=5, max_features='auto', max_leaf_nodes=None,
                            min_samples_leaf=10, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=n_est, n_jobs=1,
                            oob_score=False, random_state=None, verbose=0,
                            warm_start=False)
    correctness = cross_val_score(rf, X_train_new, y_train, cv=5)
    #error_out.append(correctness.mean())
error_out = np.array(error_out)

import matplotlib.pyplot as plt
#plt.plot(range(200), error_in, color='green', label='Error in Sample')
plt.plot(range(200), error_out, color='red', label='Error out of Sample')
plt.xlabel('n_estimators = 200, number of features is ' + str(K))
plt.ylabel('Error')
plt.show()