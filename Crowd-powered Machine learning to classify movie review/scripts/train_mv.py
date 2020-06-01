import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# get data from mturk_sample.csv
mt_sample = pd.read_csv("mturk_sample.csv")

# aggregate by majority vote
# first group on 'id' and count unique values, get 0th index which has most frequent values
mt_class = mt_sample.groupby(['id'])['class'].agg(lambda x:x.value_counts().index[0])

# take unique
del mt_sample["annotator"]
mt_sample = mt_sample.iloc[:,0:mt_sample.shape[1]-1].drop_duplicates(keep="first")

# sort values based on id.
mt_sample.sort_values("id", inplace = True)

# get training data
x_train = mt_sample.iloc[:,1:]
y_train =  mt_class.values

# import test data
test_sample_df = pd.read_csv("test.csv")
x_test = test_sample_df.iloc[:,1:-1]
y_test = test_sample_df.iloc[:,-1]

# import classifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# create object
dtc = DecisionTreeClassifier()

# train model
dtc.fit(x_train, y_train)


# predict
predict = dtc.predict(x_test)

from sklearn.metrics import classification_report
report = classification_report(y_test,predict,output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df

with open("train_mv.txt", 'w') as f:
    f.write("Classification Report\n\n")  
    f.write(report_df.to_string(header = True, index = True))

# accuracy
acc_sco = accuracy_score(y_test, predict)

# f1_score
f1_sco = f1_score(y_test, predict, average='micro')

# prediction probabilities 
propre = dtc.predict_proba(x_test)
df_propre = pd.DataFrame(propre)

# write value into text file
with open("train_mv.txt", "a") as t:
    t.write("Accuracy: " + str(acc_sco) + '\n' + "F1 Score: "+ str(f1_sco) + '\n' + "prediction probabilities: " + df_propre.to_string(header = True, index = True))