#!/usr/bin/env python
# coding: utf-8

# **PCA**

# In[3]:


#importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# In[5]:


#loading the transformed Data Set
solar_data = pd.read_csv(r"C:\Users\bathl\OneDrive\Documents\SJSU\Machine_Learning\Project\solar_data_transformed18.csv")


# In[7]:


# Exclude the target and ID column from the PCA analysis
X = solar_data.drop(['InstallType','ID'], axis=1)

# Instantiate a PCA object
pca = PCA()

# Fit the PCA object to the features of the dataframe
pca.fit(X)


# In[9]:


# Plot the explained variance ratio for each principal component
plt.plot(range(1, pca.n_components_+1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()


# In[11]:


explained_variance = pca.explained_variance_

# Calculate the percentage of explained variance of each principal component
explained_variance_ratio = explained_variance / sum(explained_variance) * 100

# Print the percentage of explained variance of each principal component
for i, variance in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {variance:.2f}%")


# We won't do PCA as it does not have high variance

# In[ ]:




