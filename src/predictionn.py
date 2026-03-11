# 1. Load scaler.pkl and model.pkl files
# 2. Create a function to predict

import pickle
import numpy as np
import os

class Insurance_Prediction:

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model_path = os.path.join(BASE_DIR, "model.pkl")

        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):

        input = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])

        input_scaled = self.scaler.transform(input)

        result = self.model.predict(input_scaled)

        return result[0]