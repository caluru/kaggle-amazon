import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Changes to seaborn theme (Comment this line to see the difference!)
sns.set()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender = pd.read_csv('gender_submission.csv')

#replace nans with -1
train['Age'].fillna(value=-20, inplace=True) 

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

#Show plots, erase history
plt.show()
