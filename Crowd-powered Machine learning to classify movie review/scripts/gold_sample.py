import pandas as pd

#Extracting 1000 samples from gold dataset
gold_dataset = pd.read_csv("gold.csv")

gold_sample = gold_dataset.sample(n=1000,random_state=19230323)

# save gold-sample to gold_sample.csv
output = 'gold_sample.csv'
gold_sample.to_csv(output, index = False)