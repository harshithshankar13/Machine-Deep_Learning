import pandas as pd
import numpy as np
import copy

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# get data from mturk_sample.csv
mt_sample = pd.read_csv("mturk_sample.csv")
mt_sample_ori = mt_sample.copy()

# aggregate by majority vote
# first group on 'id' and count unique values, get 0th index which has most frequent values
mt_class = mt_sample.groupby(['id'])['class'].agg(lambda x:x.value_counts().index[0])

# take unique
del mt_sample["annotator"]
mt_sample = mt_sample.iloc[:,0:mt_sample.shape[1]-1].drop_duplicates(keep="first")

# sort values based on id.
mt_sample.sort_values("id", inplace = True)
mt_sample["class"] = mt_class.values

# import test data
test_sample_df = pd.read_csv("test.csv")
x_test = test_sample_df.iloc[:,1:-1]
y_test = test_sample_df.iloc[:,-1]

# Dawid & Skene
mt_sample_DwSk = mt_sample_ori.iloc[:,[0, 1, -1]]

# input data    
input_DwSk = mt_sample_DwSk.pivot(index='id',columns='annotator',values='class')
input_DwSk.sort_values(by="id", inplace=True)
input_DwSk.fillna(0, inplace = True)

# Majority vote
mv_df = mt_sample.pivot(index='id',columns='class',values='class')
#mt_sample.fillna(0)

mv_df['pos'] = [1.0 if val == "pos" else 0.0 for val in mv_df['pos'].values]
mv_df['neg'] = [1.0 if val == "neg" else 0.0 for val in mv_df['neg'].values]
mv_df.reset_index(inplace=True)

for i in range(35):

    # re-initialise weights in worker list
    worker_list = list()
    for i in range(input_DwSk.shape[1]-1):
        worker_list.append([[0.0, 0.0],[0.0, 0.0]])
    
    # update worker node estimate reliability
    for col in range(1,input_DwSk.shape[1]):
        for row in range(0,input_DwSk.shape[0]):
            pos_class = mv_df["pos"].iloc[row] >= mv_df["neg"].iloc[row]
            if input_DwSk.iloc[row, col] == "pos":
                if pos_class == True:
                    worker_list[col-1][0][0] += mv_df["pos"].iloc[row]
                else:
                    worker_list[col-1][1][0] += mv_df["neg"].iloc[row]
            elif input_DwSk.iloc[row, col] == "neg":
                if pos_class == False:
                    worker_list[col-1][1][1] += mv_df["neg"].iloc[row]
                else:
                    worker_list[col-1][0][1] += mv_df["pos"].iloc[row]
    
    # Normalise on actual value in confusion matrix of worker node
    for inx in range(len(worker_list)):
        # update positive 
        tol_pos = worker_list[inx][0][0] + worker_list[inx][0][1]
        if tol_pos != 0:
            worker_list[inx][0][0] = worker_list[inx][0][0] / tol_pos
            worker_list[inx][0][1] = worker_list[inx][0][1] / tol_pos

        # update negative
        tol_neg = worker_list[inx][1][0] + worker_list[inx][1][1]
        if tol_neg != 0:
            worker_list[inx][1][0] = worker_list[inx][1][0] / tol_neg
            worker_list[inx][1][1] = worker_list[inx][1][1] / tol_neg

    # re-initialise mv_df part
    mv_df["pos"] = 0
    mv_df["neg"] = 0
            
    # re-estimate polarity estimates and update polarity
    for row in range(0,input_DwSk.shape[0]):  # 0 to 1000
        for col in range(1,input_DwSk.shape[1]):  # 1 to 185
            #print(row, col)
            if input_DwSk.iloc[row, col] == "pos":
                mv_df.at[row,"pos"] += worker_list[col-1][0][0] 
                mv_df.at[row,"neg"] += worker_list[col-1][1][0]
            elif input_DwSk.iloc[row, col] == "neg":
                mv_df.at[row,"pos"] += worker_list[col-1][0][1] 
                mv_df.at[row,"neg"] += worker_list[col-1][1][1]

    # normalise on each sentence
    for inx in range(mv_df.shape[0]):
        inx_tol = mv_df.iloc[inx].loc["pos"] + mv_df.iloc[inx].loc["neg"]
        if inx_tol != 0:
            mv_df.loc[inx,"pos"] = mv_df.iloc[inx].loc["pos"] / inx_tol
            mv_df.loc[inx,"neg"] = mv_df.iloc[inx].loc["neg"] / inx_tol
    
    
# get output in new dataframe
for row in range(mv_df.shape[0]):
    output = mv_df[["neg","pos"]].idxmax(axis = 1)

# take unique and x_train
del mt_sample_ori["annotator"]
mt_sample_ori = mt_sample_ori.iloc[:,0:mt_sample_ori.shape[1]-1].drop_duplicates(keep="first")
mt_sample_ori.sort_values("id", inplace = True)
x_train_1 = mt_sample_ori.iloc[:,1:]

# model and predict
from sklearn.tree import DecisionTreeClassifier
dtc_1 = DecisionTreeClassifier()
# build model
dtc_1.fit(x_train_1, output)

predict_1 = dtc_1.predict(x_test)


from sklearn.metrics import classification_report
report = classification_report(y_test,predict_1,output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df

with open("train_ds.txt", 'w') as f:
    f.write("Classification Report\n\n")  
    f.write(report_df.to_string(header = True, index = True))

# accuracy
acc_sco = accuracy_score(y_test, predict_1)
# accuracy
acc_sco = accuracy_score(y_test, predict_1)

# f1_score
f1_sco = f1_score(y_test, predict_1, average='micro')

# prediction probabilities 
propre = dtc_1.predict_proba(x_test)
df_propre = pd.DataFrame(propre)

# write value into text file
with open("train_ds.txt", "a") as t:
    t.write("Accuracy: " + str(acc_sco) + '\n' + "F1 Score: "+ str(f1_sco) + '\n' + "prediction probabilities: " + df_propre.to_string(header = True, index = True))