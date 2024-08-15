
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Q1
df = pd.read_csv('voters_hm4.csv')
random_seed = 123
# defining random seed

# Q2a
my_crosstab = pd.crosstab(index=df['sex'], columns=df['vote'])
print(my_crosstab)
# cross table gender with vote column
my_crosstab1 = pd.crosstab(index=df['passtime'], columns=df['vote'])
print(my_crosstab1)
# cross table hobby with vote column
my_crosstab2 = pd.crosstab(index=df['status'], columns=df['vote'])
print(my_crosstab2)
# cross table personal status with vote column

# Q2b
df.boxplot(column=['age'], by='vote', grid='false')
plt.show()
# box plot of republicans via age and democrats via age
df.boxplot(column=['salary'], by='vote', grid='false')
plt.show()
# box plot of republicans via salary and democrats via salary
df.boxplot(column=['volunteering'], by='vote', grid='false')
plt.show()
# box plot of republicans via volunteering and democrats via salary

# Q3
df.info()
# getting info on the  data frame
print(df.isna().sum())
# sum num of null values for each column
df = df.dropna(subset=['age', 'passtime'])
# dropping missing values

df['salary'] = df['salary'].replace(to_replace=np.nan, value=df.salary.median())
# replacing nan values in salary column with median
print(df.isna().sum())
# sum all null values of each column

df['SalaryNorm'] = stats.zscore(df['salary'])
# normalize salary column
df['NORMage'] = stats.zscore(df['age'])
# normalize age column
df['NORMvolunteering'] = stats.zscore(df['volunteering'])
# normalize volunteering column

gender = pd.get_dummies(df['sex'], prefix='SEXtype-')
# create gender numeric columns in a separated data frame of gender
hobby = pd.get_dummies(df['passtime'], prefix='PASStype-')
# create passtime numeric columns in a separated data frame of gender
stat = pd.get_dummies(df['status'], prefix='STATUStype-')
# create status numeric columns in a separated data frame of gender
vote = pd.get_dummies(df['vote'], prefix='VOTEtype-')
# create vote numeric columns in a separated data frame of gender
df.describe()

# prepping Q10
# <
df1 = df
# creating data frame df1 with all df vals
df1 = pd.concat([df1, gender, hobby, vote], axis=1)
# concatenate to df1 gender, hobby and vote
df1 = df.drop(['sex', 'passtime', 'vote', 'age', 'salary', 'volunteering'], axis=1)
# drop unnecessary columns (for Q10)
# >

df = pd.concat([df, gender, hobby, stat], axis=1)
# concatenate to df gender, hobby and stat
df = df.drop(['sex', 'passtime', 'status', 'age', 'salary', 'volunteering'], axis=1)
# dropping unnecessary columns

# Q4
le = preprocessing.LabelEncoder()
# converting categorical values into numeric values
le.fit(df['vote'])
# fitting the model on the results on vote column
df['target'] = le.transform(df['vote'])
# assigning the transformed numeric values into target column

features = df.drop(columns=['target', 'vote'])
# create features which is a data frame of al the features columns without the target columns
labels = df['target']
# create labels which is a data frame of target column
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=123)
# dividing the data in a 30\70 ratio and

# Q5
clf = DecisionTreeClassifier(random_state=123)
# create clf and assign to him the type of learning algorithm we would like to use
model = clf.fit(X_train, y_train)
# training the model via train data
fig = plt.figure(figsize=(14, 10))
# creating the tree figure
tree.plot_tree(model, feature_names=X_train.columns, class_names=le.classes_, filled=True)
# uses tree plot func from sklearn to plot decision tree
plt.show()

# Q6
y_test_pred = model.predict(X_test)
# creating test prediction on X_ test data
print(np.column_stack((y_test, y_test_pred)))
# combines the actual target labels and the predicted target labels and printing each one
print(confusion_matrix(y_test, y_test_pred))
# printing matrix of test prediction and test actual result
ct = pd.crosstab(y_test, y_test_pred, colnames=['pred'], margins=True)
# creating cross table of test result and test prediction and add sum column dns 0 and 1 for target column
print(ct)
# # A
print((ct.iloc[0, 0]+ct.iloc[1, 1])/(ct.iloc[2, 2]))
# calculating accuracy
# # B
print((ct.iloc[0, 0])/(ct.iloc[2, 0]))
# calculating precision
# # C
print((ct.iloc[0, 0])/(ct.iloc[0, 2]))
# calculating recall

# Q7
y_train_pred = model.predict(X_train)
# creating prediction on X_train data
ct = pd.crosstab(y_train, y_train_pred, colnames=['pred'], margins=True)
# creating cross table of y_train and t_train prediction with sum columns and target values
print(ct)
# A
print((ct.iloc[0, 0]+ct.iloc[1, 1])/(ct.iloc[2, 2]))
# calculating accuracy
# B
print((ct.iloc[0, 0])/(ct.iloc[2, 0]))
# calculating precision
# C
print((ct.iloc[0, 0])/(ct.iloc[0, 2]))
# calculating recall

# as we can see our accuracy precision and recall of the test data is ov pretty good mesures thou the train measures are
# too high at 100 percent. a stat that indicate on overfitting state

# Q8
clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=40, random_state=123)
# creating another tree only limited to depth 5 and max of 40 nodes
model = clf.fit(X_train, y_train)
# sending the model to fit on X_train, and y_train
fig = plt.figure(figsize=(14, 10))
# defining the tree figure
tree.plot_tree(model, feature_names=X_train.columns, class_names=le.classes_, filled=True)
# creating visualize for decision tree model we created
plt.show()

df.info()
print(clf.get_depth())
# depth=5
print(clf.get_n_leaves())
# num of leaf = 20
# most significant dividing feature is volunteering
# all features are included
print(df.loc[67, ['vote']])
# number 68 classified as republican, which he is republican. we followed the trail with the tree leafs

# Q9
y_test_pred = model.predict(X_test)
# get prediction of X_test on decision tree model
print(np.column_stack((y_test, y_test_pred)))
# printing the results versus the prediction
print(confusion_matrix(y_test, y_test_pred))
# printing matrix of prediction summery
ct = pd.crosstab(y_test, y_test_pred, colnames=['pred'], margins=True)
# creating a cross table of the correct results versus incorrect results with sum column and target column as 0 and 1
print(ct)
# A
print((ct.iloc[0, 0]+ct.iloc[1, 1])/(ct.iloc[2, 2]))
# calculating accuracy
# B
print((ct.iloc[0, 0])/(ct.iloc[2, 0]))
# calculating precision
# C
print((ct.iloc[0, 0])/(ct.iloc[0, 2]))
# calculating recall

y_train_pred = model.predict(X_train)
# creating prediction on X_train
ct = pd.crosstab(y_train, y_train_pred, colnames=['pred'], margins=True)
# creating cross table y_train and y_train prediction
print(ct)
# A
print((ct.iloc[0, 0]+ct.iloc[1, 1])/(ct.iloc[2, 2]))
# calculating accuracy
# B
print((ct.iloc[0, 0])/(ct.iloc[2, 0]))
# calculating precision
# C
print((ct.iloc[0, 0])/(ct.iloc[0, 2]))
# calculating recall

# Q10
# the students conducted the test on the same data which was used for the train.
le.fit(df1['status'])
# fitting the model on status column result
df1['target'] = le.transform(df1['status'])
# transforming status category column into target column with values of 0, 1 and 2

features = df1.drop(columns=['target', 'status'])
# creating features data frame of all the columns without status and target columns
labels = df1['target']
# create labels data frame which is the target column
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=123)
# dividing the data in a 70/30 ratio
clf = DecisionTreeClassifier(random_state=123)
# creating clf which is a decision tree algorithm
model = clf.fit(X_train, y_train)
# sending the model to fit on the X_train and y_train
fig = plt.figure(figsize=(14, 10))
# defining the figure of the tree
tree.plot_tree(model, feature_names=X_train.columns, class_names=le.classes_, filled=True)
# creating a plot to present the tree
plt.show()

y_test_pred = model.predict(X_test)
# creating y_test prediction on X_test
print(np.column_stack((y_test, y_test_pred)))
# printing each result versus each prediction
print(confusion_matrix(y_test, y_test_pred))
# printing confusion matrix which sum of the results of the predictions
ct = pd.crosstab(y_test, y_test_pred, colnames=['pred'], margins=True)
# creating cross table of test and prediction
print(ct)
print('accuracy ',(ct.iloc[0, 0]+ct.iloc[1, 1]+ct.iloc[2, 2])/(ct.iloc[3, 3]))
# calculating accuracy
# 0.8857142857142857

# Q11
# in order to make sure we can predict with good results the targe which is status, we need to calculate
# accuracy precision and recall train and test.
print('single test precision ', (ct.iloc[0, 0])/(ct.iloc[0, 3]))
print('family test precision ', (ct.iloc[1, 1])/(ct.iloc[1, 3]))
print('couple test precision ', (ct.iloc[2, 2])/(ct.iloc[2, 3]))
# calculating precision for single, family and couple

print('single test recall ', (ct.iloc[0, 0])/(ct.iloc[3, 0]))
print('family test recall ', (ct.iloc[1, 1])/(ct.iloc[3, 1]))
print('couple test recall ', (ct.iloc[2, 2])/(ct.iloc[3, 2]))
# calculating recall for single, family and couple

y_train_pred = model.predict(X_train)
# creating y_train prediction on X_train
print(confusion_matrix(y_train, y_train_pred))
# printing confusion matrix which sum of the results of the predictions
ct = pd.crosstab(y_train, y_train_pred, colnames=['pred'], margins=True)
# creating cross table of test and prediction
print(ct)
print('accuracy ', (ct.iloc[0, 0]+ct.iloc[1, 1]+ct.iloc[2, 2])/(ct.iloc[3, 3]))
# calculating accuracy

print('singe train precision ', (ct.iloc[0, 0])/(ct.iloc[0, 3]))
print('family train precision ', (ct.iloc[1, 1])/(ct.iloc[1, 3]))
print('couple train precision ', (ct.iloc[2, 2])/(ct.iloc[2, 3]))
# calculating precision for single, family and couple

print('singe train recall ', (ct.iloc[0, 0])/(ct.iloc[3, 0]))
print('family train recall ', (ct.iloc[1, 1])/(ct.iloc[3, 1]))
print('couple train recall ', (ct.iloc[2, 2])/(ct.iloc[3, 2]))
# calculating recall for single, family and couple

# we can see that the result of the train prediction  is too high next to the test prediction. a clear indication
# of over fitting will not predict well new data on family status

# Q12
# yes we can suspect over fitting

# Q 13
print('singe train precision ', (ct.iloc[0, 0])/(ct.iloc[0, 3]))
# calculating precision for single which is 100 percent