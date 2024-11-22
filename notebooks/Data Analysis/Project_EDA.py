#!/usr/bin/env python
# coding: utf-8

# **EDA and Feature Engineering**

# In[3]:


#importing all the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from category_encoders import HashingEncoder


# In[5]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[7]:


#loading the Data Set
solar_data = pd.read_csv(r"C:\Users\bathl\OneDrive\Documents\SJSU\Machine_Learning\Project\solar_data.csv")


# In[9]:


solar_data.info()


# In[11]:


solar_data.head(5)


# In[13]:


# Top counties with the most solar installations
# Group the data by county
county_installations = solar_data.groupby('County').size().reset_index(name='installation_count')

# Sort the data
county_installations = county_installations.sort_values(by='installation_count', ascending=False)

# Plot the top counties with the most solar installations
plt.figure(figsize=(10, 6))

sns.barplot(x='installation_count', y='County', data=county_installations.head(10), palette='Blues_d', legend=False)

plt.title('Top 10 Counties by Number of Solar Installations')
plt.xlabel('Number of Installations')
plt.ylabel('County')
plt.show()


# In[14]:


# Top counties with the most solar installations by install type
# Group the data by county and installation type
county_installations = solar_data.groupby(['County', 'InstallType']).size().reset_index(name='installation_count')

# Sort the data 
county_installations = county_installations.sort_values(by='installation_count', ascending=False)

plt.figure(figsize=(12, 8))

# Use of 'InstallType' as the hue for differentiation
sns.barplot(x='installation_count', y='County', hue='InstallType', data=county_installations.head(10), palette='Set2')

plt.title('Top 10 Counties by Number of Solar Installations (by Type)')
plt.xlabel('Number of Installations')
plt.ylabel('County')
plt.legend(title='Installation Type')
plt.show()


# In[16]:


# Plot for Urban vs Rural Installations
plt.figure(figsize=(8, 5))
sns.countplot(x='UrbanRural', data=solar_data, palette='magma')
plt.title('Urban vs Rural Installations')
plt.xlabel('Area Type')
plt.ylabel('Number of Installations')
plt.show()


# In[17]:


# Plot for Distribution of Distances to Substation 
plt.figure(figsize=(10, 6))
sns.histplot(solar_data['DistSub_100'], kde=True, color='green')
plt.title('Distribution of Distances to Substation (Within 100)')
plt.xlabel('Distance to Substation')
plt.ylabel('Density')
plt.show()


# In[19]:


# Plot for installation types
sns.countplot(x='InstallType', data=solar_data)
plt.title('Count of Different Installation Types')
plt.show()


# In[21]:


# Plot for Acres by Install Type
# Calculate mean acres by installation type
mean_acres = solar_data.groupby('InstallType')['Acres'].mean().reset_index()

# Bar plot of mean Acres by Install Type
plt.figure(figsize=(12, 6))
sns.barplot(x='InstallType', y='Acres', data=mean_acres, palette='muted')
plt.title('Average Acres by Install Type')
plt.ylabel('Average Acres')
plt.xlabel('Install Type')
plt.show()


# In[22]:


# Correlation Matrix
# Select only numeric columns for the correlation matrix
numeric_solar_data = solar_data.select_dtypes(include=[float, int])

plt.figure(figsize=(12, 8))
correlation_matrix = numeric_solar_data.corr() 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()


# Findings from Correlation Matrix: 
# 1. Acres and Area indicates perfect positive correlation of 1, suggesting these variables are the same fields but with different units.
# 2. Acres and Length: These two features are highly correlated(0.97), suggesting that they are closely related.
# 3. DistSub_100 and DistSub_200 show a moderate positive correlation of 0.65, indicating the distance to a substation at 100 and 200 units behaves similarly.
# 4. DistSub_CAISO and DistSub_200 are quite low in terms of their correlation, equating to 0.29.
# 5. Multiple pairs of variables show close to 0 correlation like "ID" and most of the rest of the features, with values close to zero, which implies negligible or zero linear relationship between pairs. 

# Since, Area and Acres are perfectly correlated, we can say that both the features bring redundacy in the dataset, and it would be better to drop one of them for the better performance and accuracy.
#     

# In[27]:


# Drop Acres as it is perfectly correlated with Area
solar_data = solar_data.drop(columns=['Acres'])


# In[29]:


solar_data.head()


# **Feature Engineering**

# In[34]:


#Creating bins for DistSub_100, DistSub_200, and DistSub_CAISO can simplify distance impacts and make it easier to capture proximity categories.
solar_data['DistSub_100_binned'] = pd.cut(solar_data['DistSub_100'], bins=[0, 1, 5, 10, 20, float('inf')], labels=['very close', 'close', 'moderate', 'far', 'very far'])
solar_data['DistSub_200_binned'] = pd.cut(solar_data['DistSub_200'], bins=[0, 1, 5, 10, 20, float('inf')], labels=['very close', 'close', 'moderate', 'far', 'very far'])
solar_data['DistSub_CAISO_binned'] = pd.cut(solar_data['DistSub_CAISO'], bins=[0, 1, 5, 10, 20, float('inf')], labels=['very close', 'close', 'moderate', 'far', 'very far'])


# In[38]:


solar_data[['DistSub_100_binned', 'DistSub_200_binned', 'DistSub_CAISO_binned']].head()


# In[30]:


#Drooping the DistSub fields, as the above new three binned features are created from these features
solar_data = solar_data.drop(columns=['DistSub_100','DistSub_200','DistSub_CAISO'])


# In[32]:


#Lable encoding the target 'InstallType' feature
label_encoder = LabelEncoder()
solar_data['InstallType'] = label_encoder.fit_transform(solar_data['InstallType'])


# In[33]:


solar_data.head()


# In[34]:


# Separate numerical and categorical features
numerical_features = solar_data.select_dtypes(include=['int64', 'float64']).columns.drop('ID','InstallType').tolist()
categorical_features = solar_data.select_dtypes(include=['object', 'category']).columns.tolist()
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# **Hash Encoding to the categorical features**

# In[39]:


pip install category_encoders


# In[46]:


hash_encoder = HashingEncoder(cols=categorical_features, n_components=13)  # 13-bit hash
solar_data[categorical_features] = hash_encoder.fit_transform(solar_data[categorical_features])


# In[47]:


solar_data.head()


# **Scaling: StandardScaler**
# 
# Scaling the numerical variables so that they have a mean of 0 and a standard deviation of 1

# In[49]:


scaler = StandardScaler()

solar_data[numerical_features] = scaler.fit_transform(solar_data[numerical_features])


# In[50]:


solar_data.head()


# In[51]:


solar_data.info()


# In[58]:


solar_data.to_csv(r"C:\Users\bathl\OneDrive\Documents\SJSU\Machine_Learning\Project\solar_data_transformed18.csv", index=False)


# In[ ]:





# In[ ]:




