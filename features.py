import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gaze_df = pd.read_csv('features.csv')

def add_HOO_count_latency(all_AOI_hit, aoi_col="AOI_HOSide"):
    AOI = aoi_col[4:] if len(aoi_col) > 4 else aoi_col
    print("AOI:", AOI)

    def summarize_group(gaze_df):
        hits = gaze_df[gaze_df[aoi_col]]
        gaze_count = len(hits)
        gaze_first = hits['timestamp'].min() if gaze_count > 0 else np.nan
        return pd.Series({
            f"{AOI}_GazeCount": gaze_count,
            f"{AOI}_FirstLook": gaze_first
        })

    summary = (
        all_AOI_hit
        .groupby(['Participant_anon', 'Problem_id'], group_keys=False)
        .apply(summarize_group)
        .reset_index()
    )

    all_AOI_hit = all_AOI_hit.merge(summary, on=['Participant_anon', 'Problem_id'], how='left')
    return all_AOI_hit


gaze_df = add_HOO_count_latency(gaze_df)
print(gaze_df['HOSide_GazeCount'].head())
