#!/usr/bin/env python
# coding: utf-8

# # DATA PREPROCESSING - Predicting Solar Installations for Sustainability and Ecological Impact 

# In[32]:


#Team members:Bhavana Meravanige Veerappa, Raksha Ravishankar, Sonia Bathla
#importing all the necessary libraries
from scipy.stats import zscore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[33]:


#loading the Data Set
solar_data = pd.read_csv(r"C:\HOME\SJSU\Solar_Data\Solar_Footprints_V2_5065925295652909767.csv")


# In[34]:


solar_data.head(2)


# In[35]:


#Encoding the Labels
solar_data = solar_data.rename(columns={
    'OBJECTID': 'ID',
    'County': 'County',
    'Acres': 'Acres',
    'Install Type': 'InstallType',
    'Urban or Rural': 'UrbanRural',
    'Combined Class': 'Class',
    'Distance to Substation (Miles) GTET 100 Max Voltage': 'DistSub_100',
    'Percentile (GTET 100 Max Voltage)': 'Percent_100',
    'Substation Name GTET 100 Max Voltage': 'Substation_100',
    'HIFLD ID (GTET 100 Max Voltage)': 'HIFLD_100',
    'Distance to Substation (Miles) GTET 200 Max Voltage': 'DistSub_200',
    'Percentile (GTET 200 Max Voltage)': 'Percent_200',
    'Substation Name GTET 200 Max Voltage': 'Substation_200',
    'HIFLD ID (GTET 200 Max Voltage)': 'HIFLD_200',
    'Distance to Substation (Miles) CAISO': 'DistSub_CAISO',
    'Percentile (CAISO)': 'Percent_CAISO',
    'Substation CASIO Name': 'Substation_CAISO',
    'HIFLD ID (CAISO)': 'HIFLD_CAISO',
    'Solar Technoeconomic Intersection': 'SolarTech',
    'Shape__Area': 'Area',
    'Shape__Length': 'Length'
})


# In[36]:


solar_data.info()


# # Handling Null Values

# In[37]:


#finding empty entries
solar_data.isnull().sum()


# In[38]:


#heatmap to showcase null values
plt.figure(figsize=(5, 5))
sns.heatmap(solar_data.isnull(), cbar=False)

#Adding the labels
plt.title('Heatmap of Missing Values')
plt.xlabel('Features')
plt.ylabel('Data Points')

#Showing the plot
plt.show()


# In[39]:


#Imputing unknown ID values-HIFLD_100, HIFLD_200, HIFLD_CAISO to 0 as the fields are in Int
solar_data['HIFLD_100'].fillna(0, inplace=True)
solar_data['HIFLD_200'].fillna(0, inplace=True)
solar_data['HIFLD_CAISO'].fillna(0, inplace=True)


# In[40]:


#Imputing unknown Substation_CAISO as 'NaN'
solar_data['Substation_CAISO'].fillna('NaN', inplace=True)


# In[41]:


solar_data.isnull().sum()


# # Checking for Duplicate Entries

# In[42]:


#checking for duplicate entries
duplicates = solar_data[solar_data.duplicated(subset='ID', keep=False)]


# In[43]:


duplicates


# # Checking for Unique Entries in Categorical Columns for the Data Consistency

# In[44]:


County = solar_data['County'].unique()
County


# In[45]:


#Removing 'County' keyword in County column
solar_data['County'] = solar_data['County'].str.replace(' County', '', regex=False)


# In[46]:


InstallType = solar_data['InstallType'].unique()
InstallType


# In[47]:


UrbanRural =solar_data['UrbanRural'].unique()
UrbanRural


# In[48]:


Class =solar_data['Class'].unique()
Class


# In[49]:


Percent_100 = solar_data['Percent_100'].unique()
Percent_100


# In[50]:


#Replacing the percentile to easily identifiable labels
replace_dict = {
    '0 to 25th': '0-25',
    '25th to 50th': '25-50',
    '50th to 75th': '50-75',
    '75th to 100th': '75-100'
}

#Replacing multiple items using replace_dict
solar_data['Percent_100'] = solar_data['Percent_100'].replace(replace_dict)
solar_data['Percent_200'] = solar_data['Percent_200'].replace(replace_dict)
solar_data['Percent_CAISO']=solar_data['Percent_CAISO'].replace(replace_dict)


# In[51]:


Substation_100 = solar_data['Substation_100'].unique()
Substation_100


# In[52]:


Substation_200 = solar_data['Substation_200'].unique()
Substation_200


# In[53]:


Substation_CAISO = solar_data['Substation_CAISO'].unique()
Substation_CAISO


# In[54]:


SolarTech = solar_data['SolarTech'].unique()
SolarTech


# In[55]:


#Rounding off to 3 decimal places-Mostly would continue with the available data as it showcase original precision.
#solar_data['Acres'] = solar_data['Acres'].round(3)
#solar_data['DistSub_100'] = solar_data['DistSub_100'].round(3)
#solar_data['DistSub_CAISO'] = solar_data['DistSub_CAISO'].round(3)
#solar_data['Area']= solar_data['Area'].round(3)
#solar_data['Length']= solar_data['Length'].round(3)


# In[56]:


#changing the float data type of HIFLD to Int data types as they are ID fields.
solar_data['HIFLD_100'] = solar_data['HIFLD_100'].astype(int)
solar_data['HIFLD_200'] = solar_data['HIFLD_200'].astype(int)
solar_data['HIFLD_CAISO'] = solar_data['HIFLD_CAISO'].astype(int)

#converting int64 to int32 for ID field:
solar_data['ID'] = solar_data['ID'].astype('int32')


# In[57]:


solar_data.head(3)


# # Outlier Analysis

# ### The first step is to find plot the distribution of the numerical features.

# In[58]:


#defining the numerical columns:
numeric_columns = ["Acres", "DistSub_100", "HIFLD_100", "DistSub_200", "HIFLD_200",
                   "DistSub_CAISO", "HIFLD_CAISO", "Area", "Length"]


# In[59]:


#plots for understanding how the distributions are:
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()

#histogram
for i, col in enumerate(numeric_columns):
    axes[i].hist(solar_data[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(f'{col}')

plt.tight_layout()
plt.show()


# From the above plot we can understand that the distributions of the numerical features are skewed. In this case, we can find the outliers by IQR method as this method is less sensitive to skweness.

# ### Finding out how many outliers are present in the features.

# In[60]:


#Initializing a dictionary to store outlier indices for each column
outliers_dict = {}

for col in numeric_columns:
    #Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
    Q1 = solar_data[col].quantile(0.25)
    Q3 = solar_data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    #Defining the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    #Identifying the outliers
    outliers = solar_data[(solar_data[col] < lower_bound) | (solar_data[col] > upper_bound)].index
    outliers_dict[col] = outliers.tolist()  # Store indices of outliers for each column

    print(f"Outliers in {col}: {len(outliers)} values")

#Displaying the outliers dictionary
#print(outliers_dict)


# ### Box plots for visual presentation of the outliers:

# In[61]:


#plotting the box plots for visual presentation of the outliers:
import matplotlib.pyplot as plt
import seaborn as sns

columns = solar_data.columns  #column names

#defining the figure size and layout
plt.figure(figsize=(15, 14))

#Loop through each column and plot individually
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(4, 4, i)  #Adjust layout
    sns.boxplot(data=solar_data[column])
    plt.title(column)

#tight_layout
plt.tight_layout()
plt.show()


# As there are many number of outliers, if we remove the records, with outliers there would be data loss. So, we are considering to cap the outliers to fall within 1.5 times the IQR from the first and third quartiles.

# In[62]:


#capping the outliers by restricting values to within 1.5 times the IQR from the first and third quartiles
for col in numeric_columns:
    Q1 = solar_data[col].quantile(0.25)
    Q3 = solar_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    solar_data[col] = solar_data[col].clip(lower=lower_bound, upper=upper_bound)


# ### Checking if all the outliers are handled.

# In[63]:


outliers_dict = {}

for col in numeric_columns:
    Q1 = solar_data[col].quantile(0.25)
    Q3 = solar_data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    #Defining the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    #Identifying outliers
    outliers = solar_data[(solar_data[col] < lower_bound) | (solar_data[col] > upper_bound)].index
    outliers_dict[col] = outliers.tolist()  # Store indices of outliers for each column

    print(f"Outliers in {col}: {len(outliers)} values")

#Displaying the outliers dictionary
#print(outliers_dict)


# ### Box plot after the outlier removal:

# In[64]:


#plotting the box plots for visual presentation of the outliers:
import matplotlib.pyplot as plt
import seaborn as sns

columns = solar_data.columns  #column names

#defining the figure size and layout
plt.figure(figsize=(15, 25))

#Loop through each column and plot individually
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(4, 4, i)  #Adjust layout
    sns.boxplot(data=solar_data[column])
    plt.title(column)

#tight_layout
plt.tight_layout()
plt.show()


# ### Cleaned Solar_data Dataset

# In[65]:


solar_data


# In[66]:


#loading the dataframe as csv file to the local
solar_data.to_csv('C:\HOME\SJSU\Solar_Data\solar_data.csv', index=False) 


# In[ ]:





# In[ ]:




