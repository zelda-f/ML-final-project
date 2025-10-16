import pandas as pd

gaze_df = pd.read_csv("gaze_all_anon.csv")

behavioral_df = pd.read_csv("behavioral_all_anon.csv")

gaze_mask = gaze_df['Problem_id'].str.contains("NSNC")
gaze_mask = gaze_mask.fillna(False)

behavioral_mask = behavioral_df['Problem_id'].str.contains("NSNC")
behavioral_mask = behavioral_mask.fillna(False)


gaze_df = gaze_df[gaze_mask]

behavioral_df = behavioral_df[behavioral_mask]


gaze_df.to_csv("gaze_all_anon.csv")
behavioral_df.to_csv("behavioral_all_anon.csv")