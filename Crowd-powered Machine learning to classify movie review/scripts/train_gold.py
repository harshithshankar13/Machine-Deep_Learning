import pandas as pd

gold_sample = pd.read_csv("gold_sample.csv")

features=gold_sample.iloc[:,1:-1]
features

labels=gold_sample.iloc[:,-1]
labels

test_dataset = pd.read_csv("test.csv")
test_features = test_dataset.iloc[:,1:-1]
test_labels = test_dataset.iloc[:,-1]

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(features, labels)
pred_labels = classifier.predict(test_features)

from sklearn.metrics import classification_report
report = classification_report(test_labels,pred_labels,output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df

with open("train_gold.txt", 'a') as f:
    f.write("Classification Report\n\n")  
    f.write(report_df.to_string(header = True, index = True))
    
predict_proba = classifier.predict_proba(test_features)

df_predict_proba = pd.DataFrame(predict_proba)
with open("train_gold.txt", 'a') as f:
    f.write("\n\nPrediction Probabilities\n\n")  
    f.write(df_predict_proba.to_string(header = True, index = True))
