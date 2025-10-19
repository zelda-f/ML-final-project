# Where are solvers looking at a given time stamp?
# or Where are solvers looking when have solved x% of the problem (in time)?
# Success rate depending on initial gaze?
# Response time correlated to correct rate?
# look at left to right tendency and how HOO placement effects performance through gaze?
# 1. HOO placement --> performance
# 2. HOO place + initial gaze location --> performance
# 3. HOO place * (initial gaze location + some other gaze metric) --> performance

# filter out color and spacing conditions
# left with NC and NS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_gaze = pd.read_csv('gaze_all_anon.csv')

df_perf = pd.read_csv('behavioral_all_anon.csv')

print(df_perf['TaskCorrect'].mean())

df_gaze['timestamp'].hist(bins=60, alpha=0.7)
plt.title("Histogram of 'Timestamps'")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()

df_gaze['x'].hist(bins=60, color='orange', alpha=0.7)
plt.title("Histogram of 'x locations'")
plt.xlabel("x location")
plt.ylabel("Frequency")
plt.show()

df_gaze['y'].hist(bins=60, color='purple', alpha=0.7)
plt.title("Histogram of 'y locations'")
plt.xlabel("y location")
plt.ylabel("Frequency")
plt.show()
