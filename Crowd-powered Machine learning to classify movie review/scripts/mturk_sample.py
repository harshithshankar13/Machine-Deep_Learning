import pandas as pd

# Import from mturk.csv and from gold_sample
mturk_df = pd.read_csv("mturk.csv")
gold_sample_df = pd.read_csv("gold_sample.csv")

# get mturk row on matching ids of mturk and gold_sample.
mt_sample = mturk_df.loc[mturk_df['id'].isin(gold_sample_df["id"])]

# save mt_sample to mturk_sample file
mt_sample.to_csv("mturk_sample.csv", index=False)
