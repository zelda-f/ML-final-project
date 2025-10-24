# Where are solvers looking at a given time stamp?
# or Where are solvers looking when have solved x% of the problem (in time)?
# Success rate depending on initial gaze?
# Response time correlated to correct rate?
# look at left to right tendency and how HOO placement effects performance through gaze?
# 1. HOO placement --> performance
# 2. HOO place + initial gaze location --> performance
# 3. HOO place * (initial gaze location + some other gaze metric) --> performance

# suiloette score to find number of cluster
# variance
# HOO look first?
# include covariates (problems ID) 
# how long they looked at HOO on average
# multiplication vs division 
# regulatization --> one hot drop one so you dont have multi colineararty

# run a correlation matrix to avoid two correlated features


# 2. causal inference (LOOP)
# 1. feature engineering
# 3. run model 

# Random forest model (fancy)
# boot strap -- (resample with replacement)
# train random forest with all features and missing a feature 
# two distribution and then we can say that the feature is significant 

# google: how can we check for feature significance in a non linear model?





# filter out color and spacing conditions
# left with NC and NS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_gaze = pd.read_csv('gaze_all_anon.csv')

df_perf = pd.read_csv('behavioral_all_anon.csv')

print(df_perf['TaskCorrect'].mean())

# df_gaze['timestamp'].hist(bins=60, alpha=0.7)
# plt.title("Histogram of 'Timestamps'")
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.show()

# df_gaze['x'].hist(bins=60, color='orange', alpha=0.7)
# plt.title("Histogram of 'x locations'")
# plt.xlabel("x location")
# plt.ylabel("Frequency")
# plt.show()

# df_gaze['y'].hist(bins=60, color='purple', alpha=0.7)
# plt.title("Histogram of 'y locations'")
# plt.xlabel("y location")
# plt.ylabel("Frequency")
# plt.show()

joint_df = pd.merge(df_gaze, df_perf, on = ['Problem_id', 'Participant_anon'], how='left')

to_drop = ['Unnamed: 0.3','Unnamed: 0.2_x','Unnamed: 0.1_x','Unnamed: 0_x', 'Unnamed: 0.2_y','Unnamed: 0.1_y','Unnamed: 0_y', 'bid']
joint_df = joint_df.drop(to_drop, axis=1)
joint_df.to_csv('features.csv')



