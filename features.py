import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


gaze_df = pd.read_csv('features.csv')

# Add time to first gaze and gaze count

def add_HOO_count_latency(all_AOI_hit, aoi_col="AOI_HOSide"):
    """
    Adds per-participant, per-problem gaze summary columns to the DataFrame
    based on a boolean AOI column.
    
    Creates:
      - <AOI>_GazeCount: number of True values per group
      - <AOI>_GazeLatency: first timestamp where AOI is True
    
    Modifies the DataFrame in place.
    """
    # Extract AOI name suffix (like R's substr(checked_AOI, 5, nchar(...)))
    AOI = aoi_col[4:] if len(aoi_col) > 4 else aoi_col

    def summarize_group(gaze_df):
        hits = gaze_df[gaze_df[aoi_col]]
        gaze_count = len(hits)
        gaze_first = hits['timestamp'].min() if gaze_count > 0 else np.nan
        return pd.Series({
            f"{AOI}_GazeCount": gaze_count,
            f"{AOI}_FirstLook": gaze_first
        })

    # Compute summary per participant/problem
    summary = (
        all_AOI_hit
        .groupby(['Participant', 'Problem_id'], group_keys=False)
        .apply(summarize_group)
        .reset_index()
    )

    # Merge back into the original dataframe
    all_AOI_hit.merge(summary, on=['Participant', 'Problem_id'], how='left', inplace=True)


gaze_df = add_HOO_count_latency(gaze_df)
print(gaze_df['AOI_HOSide_GazeCount'].head())

