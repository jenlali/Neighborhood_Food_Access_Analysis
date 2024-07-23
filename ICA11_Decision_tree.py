#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


# In[4]:


INPUT_FILENAME    = "NeighborhoodFood.csv"
TRAINING_PART     = 0.60
MAX_DEPTH         = 4
MINIMUMSPLIT      = 45
OUTPUT_COLUMN     = 'ACCESS'


# In[5]:


df = pd.read_csv(INPUT_FILENAME)
df = df.dropna(axis=0, how='any')

features = df.drop(columns = ['OBJECTID', OUTPUT_COLUMN])
target = df[OUTPUT_COLUMN]
print(features)


# In[12]:


dummyFeatures = pd.get_dummies(features)


# In[13]:


#splitting the dataset into a training and testing set
xTrain,xTest,yTrain,yTest = train_test_split(dummyFeatures, target, train_size = TRAINING_PART, random_state = 0, stratify = target)

#parameters for decision tree
dTree = DecisionTreeClassifier(max_depth = MAX_DEPTH, min_samples_split = MINIMUMSPLIT, random_state = 0)

#fitting the tree to the training model
dTree.fit(xTrain, yTrain)

featureNames = list(dummyFeatures.columns)

fig, ax = plt.subplots(figsize = (40,20))
tree.plot_tree(dTree, node_ids = True, proportion = True, impurity = False, fontsize=20, feature_names = featureNames, class_names = ['0','1'], rounded = True, filled = True)
plt.show()


# In[8]:


#Getting predictions based on training and test sets
yTrainPred = dTree.predict(xTrain)
yTestPred = dTree.predict(xTest)

#evaluating the accuracy of each
trainAccuracy = accuracy_score(yTrainPred, yTrain)
testAccuracy = accuracy_score(yTestPred, yTest)
print(trainAccuracy, testAccuracy)


# In[9]:


# Generating Confusion Matrices for the training set:
predicted = yTrainPred
observed = yTrain
confusionMatrix = confusion_matrix(observed, predicted)

print(confusionMatrix)


# In[10]:


# Generating Confusion Matrices for the validation set:
predictedVal = yTestPred
observedVal = yTest
confusionMatrixVal = confusion_matrix(observedVal, predictedVal)
 
print(confusionMatrixVal)


# In[11]:


# Correct Classification Rate:
# Check whether there is a match between each predicted value (in pred) and the actual value
predRateTraining = mean(yTrainPred == yTrain)
predRateValidation = mean(yTestPred == yTest)
trainingPercentage = "{:.2%}".format(predRateTraining)
validationPercentage = "{:.2%}".format(predRateValidation)

print("The correct classification rate based on the training set is " + trainingPercentage)
print("The correct classification rate based on the validation set is " + validationPercentage)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




