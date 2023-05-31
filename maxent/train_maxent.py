import pandas as pd
import numpy as np
from maxent.maxent import MaxEnt
from sklearn.metrics import accuracy_score

train_data_path = 'yelp_review_polarity_csv/train.csv'
train_data_df = pd.read_csv(train_data_path, header=None)
train_texts = train_data_df.iloc[:, 1].tolist()
train_labels = train_data_df.iloc[:, 0].tolist()

test_data_path = 'yelp_review_polarity_csv/test.csv'
test_data_df = pd.read_csv(test_data_path, header=None)
test_texts = test_data_df.iloc[:, 1].tolist()
test_labels = test_data_df.iloc[:, 0].tolist()

train_labels = np.array(train_labels) - 1
test_labels = np.array(test_labels) - 1

model = MaxEnt(M=100)
model.fit(train_texts, train_labels, test_texts, test_labels)

pred = model.predict(test_texts)
print(pred)
print(accuracy_score(test_labels, pred))