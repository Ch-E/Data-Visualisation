# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 09:18:12 2019

@author: Charl
"""

#%% Import data

import pandas as pd
import numpy as np

#%% Read dataset

train = pd.read_csv('D:/Documents/UFS/5th Year/Honours Project/Data Sources/train_V2.csv')

train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

# Create distance feature
train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train.drop(['rideDistance','walkDistance','swimDistance'],inplace=True,axis=1)

# Create headshot_rate feature
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)

# Create playersJoined feature - used for normalisation
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

#%% Data cleaning - removing outliers

# Row with NaN 'winPlacePerc' value - pointed out by averagemn (https://www.kaggle.com/donkeys)
train.drop(2744604, inplace=True)

# Players who got kills without moving
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['distance'] == 0))
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

# Players who got more than 10 roadkills
train.drop(train[train['roadKills'] > 10].index, inplace=True)

# Players who made a minimum of 9 kills and have a headshot_rate of 100%
train[(train['headshot_rate'] == 1) & (train['kills'] > 8)].head(10)

# Players who made kills with a distance of more than 1 km
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

# Players who acquired more than 80 weapons
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

# Players how use more than 40 heals
train['heals'] = train['boosts']+train['heals']
train.drop(train[train['heals'] >= 40].index, inplace=True)

# Create normalised features
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['assistsNorm'] = train['assists']*((100-train['playersJoined'])/100 + 1)
train['roadKillsNorm'] = train['roadKills']*((100-train['playersJoined'])/100 + 1)
train['vehicleDestroysNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['killPointsNorm'] = train['vehicleDestroys']*((100-train['playersJoined'])/100 + 1)
train['headshotKillsNorm'] = train['headshotKills']*((100-train['playersJoined'])/100 + 1)
train['revivesNorm'] = train['revives']*((100-train['playersJoined'])/100 + 1)

# Features that will be used for training
predictors = [
              "numGroups",
              "distance",
              "boosts",
              "killStreaks",
              "DBNOs",
              "killPlace",
              "killStreaks",
              "longestKill",
              "heals",
              "weaponsAcquired",
              "headshot_rate",
              "assistsNorm",
              "headshotKillsNorm",
              "damageDealtNorm",
              "killPointsNorm",
              "revivesNorm",
              "roadKillsNorm",
              "vehicleDestroysNorm",
              "killsNorm",
              "maxPlaceNorm",
              "matchDurationNorm",
              ]

# Id is an insignificant feature, because it isn't in the test set

X = train[predictors]
X.head()

y = train['winPlacePerc']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#%% Model - RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

RFtree = RandomForestRegressor(random_state=101, criterion="mse", max_depth=14, max_features=None,
                               max_leaf_nodes=None, min_samples_leaf=4, min_samples_split=14, n_estimators=100, n_jobs=-1)

RFtree.fit(X_train, y_train)

predictions = RFtree.predict(X_test)

#%% Model - LightGBM

import lightgbm as lgb

params = {
            'num_leaves': 144,
            'learning_rate': 0.1,
            'n_estimators': 900,
            'max_depth': 12,
            'max_bin': 55,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'feature_fraction': 0.9
         }



#LightGBM parameters
lgbm_reg = lgb.LGBMRegressor(num_leaves=params['num_leaves'], learning_rate=params['learning_rate'],
                             n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                             max_bin = params['max_bin'], bagging_fraction = params['bagging_fraction'],
                             bagging_freq = params['bagging_freq'], feature_fraction = params['feature_fraction'],
                            )

lgbm_reg.fit(X_train, y_train)

predictions2 = lgbm_reg.predict(X_test)

#%% Evaluation

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

MAE = mean_absolute_error(y_test, predictions2)
MSE = mean_squared_error(y_test, predictions2)
R2 = r2_score(y_test, predictions2)

print("Metrics:")
print("-------------------------------")
print("Mean Absolute Error: {}".format(MAE))
print("Mean Squared Error: {}".format(MSE))
print("R2 Score: {}".format(R2))

#%% Data visualisation

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Matrix
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

# Feature Importance
import lightgbm as lgb
lgb.plot_importance(lgbm_reg, max_num_features=20, figsize=(10, 8));
plt.title('Feature importance');
