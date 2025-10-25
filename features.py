import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gaze_df = pd.read_csv('features.csv')

def add_gaze_count_latency(all_AOI_hit, aoi_col="AOI_HOSide"):
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

gaze_df = add_gaze_count_latency(gaze_df)
gaze_df = add_gaze_count_latency(gaze_df, aoi_col="AOI_LOSide")
print(gaze_df['HOSide_GazeCount'].head())
print(gaze_df['LOSide_GazeCount'].head())

# Add time quartiles

def assign_quartile(group):
    """
    Adds column "time_quartile" which is an int 1-4 that represents the time quartile of the timestamp column
    """
    total_time = group['task_resp.rt'].iloc[0]
    q1, q2, q3 = total_time / 4, total_time / 2, 3 * total_time / 4

    def get_quartile(ts):
        if ts <= q1:
            return 1
        elif ts <= q2:
            return 2
        elif ts <= q3:
            return 3
        else:
            return 4

    group['time_quartile'] = group['timestamp'].apply(get_quartile)
    return group

gaze_df = gaze_df.groupby(['Participant_anon', 'Problem_id'], group_keys=False).apply(assign_quartile)

print(gaze_df['time_quartile'].head())