from __future__ import division

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Changes to seaborn theme (Comment this line to see the difference!)
sns.set()

#Read in data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender = pd.read_csv('gender_submission.csv')

#replace nans (Not sure if this is the best way to deal with them, some have a lot of nans
train['Age'].fillna(value=-20, inplace=True) #177 / 891 are nan
train['Cabin'].fillna(value='Z', inplace=True) #687 / 891 are nan...

#separate training data for living and dead people
lived = train.loc[train['Survived'] == 1]
died = train.loc[train['Survived'] == 0]

#Histograms of age among all data, survivors, and deceased
#All
sns.distplot(train['Age'])
plt.title('Age Distribution of all passengers (Dead or Alive)')
plt.figure() #Remove both plt.figure() commands to superimpose histograms

#Living
sns.distplot(lived['Age'])
plt.title('Age Distribution of Living Passengers')
plt.figure()

#Deceased
sns.distplot(died['Age'])
plt.title('Age Distribution of Dead Passengers')

print "Ratio of survivors to total passengers by age"

boundaries = range(0,int(max(train['Age'])) + 10, 5)

for i in range(len(boundaries) - 1):
	lower = boundaries[i]
	upper = boundaries[i+1]
	live = len(lived[(lived.Age >= lower) & (lived.Age < upper)])
	total = len(train[(train.Age >= lower) & (train.Age < upper)])
	if total > 0:
		print (boundaries[i], boundaries[i+1]), live / total

print "----------------------------------------------------"

#Show plots, erase history
plt.show()

#Looking at effect of ticket class on survival rates
sns.countplot(train['Pclass'])
plt.title('Class distribution of all passengers')
plt.figure()

sns.countplot(lived['Pclass'])
plt.title('Class distribution of surviving passengers')
plt.figure()

sns.countplot(died['Pclass'])
plt.title('Class distribution of deceased passengers')

plt.show()

print "Ratio of survived to total num passengers for each ticket class"

for cls in [1,2,3]:
	print cls, len(lived.loc[lived['Pclass'] == cls]) / len(train.loc[train['Pclass'] == cls])

#Cabin information
