# 1. load scaler.pkl and model.pkl files
# 2. create a function to predict


import os
import pickle
import numpy as np

class Insurance_Prediction:
    def __init__(self):

        base_path = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base_path, "..", "artifacts", "model.pkl")
        scaler_path = os.path.join(base_path, "..", "artifacts", "scaler.pkl")

        self.model = pickle.load(open(model_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))

    def prediction(self,Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        input=np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])
        scaled_input=self.scaler.transform(input)
        result=self.model.predict(scaled_input)
        return result[0]