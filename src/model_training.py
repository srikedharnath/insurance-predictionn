# 1. load processed data from processed folder
# 2. create model and train data
# 3. save model in artifacts folder
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

X_train=pd.read_csv("../data/processed/X_train.csv")
X_test=pd.read_csv("../data/processed/X_test.csv") 
y_train=pd.read_csv("../data/processed/y_train.csv")
y_test=pd.read_csv("../data/processed/y_test.csv")

model = LinearRegression()
model.fit(X_train, y_train)

with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)

print("Successfully saved your model file")
