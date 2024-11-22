#!/usr/bin/env python
# coding: utf-8

# #### Loading the pickled file for demo (Example Code)

# In[4]:


import pandas as pd
import pickle

#Loading the pickled model
model_path = r"C:\Users\bhava\Downloads\LogisticRegression.pkl"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully.")



# In[5]:


#Define feature names (should match the training data columns)
feature_names = [
    "County", "InstallType", "UrbanRural", "Percent_100", "Substation_100",
    "HIFLD_100", "Percent_200", "Substation_200", "HIFLD_200",
    "Percent_CAISO", "Substation_CAISO", "HIFLD_CAISO", "SolarTech",
    "Area", "Length", "DistSub_100_binned", "DistSub_200_binned", "DistSub_CAISO_binned"
]
class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

#Example input as a DataFrame
sample_input = pd.DataFrame(
    [[3, 2, 0, 0, 0, 1.078413738, 0, 2, 1.294543729, 3, 0, 1.589278262, 0, -0.534259442, -0.455304438, 2, 3, 0]],
    columns=feature_names
)


# In[3]:


#Predict the class
predicted_class = loaded_model.predict(sample_input)[0]

#Predict probabilities
probabilities = loaded_model.predict_proba(sample_input)[0]

#Get the real name for the predicted class
predicted_name = class_mapping[predicted_class]


print(f"Input: {sample_input.iloc[0].values}")
print(f"Predicted Class Index: {predicted_class}")
print(f"Predicted Class Name: {predicted_name}")
print(f"Prediction Probabilities: {probabilities}")


# In[ ]:




