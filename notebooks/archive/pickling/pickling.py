#!/usr/bin/env python
# coding: utf-8

# #### Pickling Example Code

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

#Loading the dataset
data_path = r"C:\HOME\SJSU\Solar_Data\solar_data_transformed18.csv"
data = pd.read_csv(data_path)


# In[2]:


#Selecting the feautures and defining the target variables
X = data.drop(columns=["ID", "Class"])
y = data["Class"]


# In[3]:


#Data Splitting into Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Saving (pickling) the trained model
model_path = r"C:\Users\bhava\Downloads\LogisticRegression.pkl"  # Update the path if needed
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully.")


# In[ ]:




