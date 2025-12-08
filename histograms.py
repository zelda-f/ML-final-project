# another data cleaning step

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_gaze = pd.read_csv('gaze_all_anon.csv')

df_perf = pd.read_csv('behavioral_all_anon.csv')

print(df_perf['TaskCorrect'].mean())


joint_df = pd.merge(df_gaze, df_perf, on = ['Problem_id', 'Participant_anon'], how='left')

to_drop = ['Unnamed: 0.3','Unnamed: 0.2_x','Unnamed: 0.1_x','Unnamed: 0_x', 'Unnamed: 0.2_y','Unnamed: 0.1_y','Unnamed: 0_y', 'bid']
joint_df = joint_df.drop(to_drop, axis=1)
joint_df.to_csv('features.csv')



