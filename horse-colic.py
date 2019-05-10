# https://www.kaggle.com/uciml/horse-colic

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

file_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data'
file_path_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test'
# file_path = 'horse-colic.data'
# file_path_test = 'horse-colic.test'
header = ['surgery', 'age', 'hospital_number', 'rectal_temp', 'pulse', 'respiratory_rate', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'nasogastric_reflux_ph', 'rectal_exam_feces', 'abdomen', 'packed_cell_volume', 'total_protein', 'abdomo_appearance', 'abdomo_protein', 'outcome', 'surgical_lesion', 'lesion_1', 'lesion_2', 'lesion_3', 'cp_data']

# training set
df_train = pd.read_csv(file_path, delim_whitespace=True, header=None)
df_train.replace("?", np.NaN,inplace=True)
df_train = df_train.astype(np.float)

df_train = df_train.dropna(subset=[22])
df_train.columns = header

# test set
df_test = pd.read_csv(file_path_test, delim_whitespace=True, header=None)
df_test.replace("?", np.NaN,inplace=True)
df_test = df_test.astype(np.float)

df_test = df_test.dropna(subset=[22])
df_test.columns = header

# features & target
y_train = df_train.outcome
X_train = df_train.drop(['hospital_number', 'respiratory_rate', 'outcome'], axis=1)

y_test = df_test.outcome
X_test = df_test.drop(['hospital_number', 'respiratory_rate', 'outcome'], axis=1)

model_v1 = XGBClassifier(learning_rate=0.01, n_estimators=500)
model_v1.fit(X_train, y_train)

# make prediction
y_pred = model_v1.predict(X_test)
predictions = [round(value) for value in y_pred]

# measure
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))