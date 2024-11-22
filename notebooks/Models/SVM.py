#!/usr/bin/env python
# coding: utf-8

# #### Support Vector Machine - Model

# In[1]:


#importing the libraries
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


#Loading the transformed and standardized Data
import pandas as pd
data = pd.read_csv(r"C:\HOME\SJSU\Solar_Data\solar_data_transformed18.csv")


# In[3]:


#Selecting the features and defining the target features
X = data.drop(columns=['InstallType', 'ID'])
y = data['InstallType']


# In[4]:


#Splitting the data into Train and Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[5]:


#Training the SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)


# In[6]:


#predictions
y_pred = svm_model.predict(X_test)


# In[7]:


#evaluation
accuracy = svm_model.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy}")

#Confusion Matrix
print("\nConfusion Matrix for SVM:\n", confusion_matrix(y_test, y_pred))


# In[8]:


from sklearn.metrics import classification_report

class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

target_names = [class_mapping[label] for label in sorted(class_mapping.keys())]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))


# #### Hyperparameter Tuning for Support Vector Machine

# In[9]:


#Defining the Hyperparameter grid for SVM
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],  #Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  #Kernel type
    'gamma': ['scale', 'auto']  #Kernel coefficient
}

grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


# In[10]:


#Best parameters and model performance
svm_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# In[11]:


#Predictions
y_pred = svm_model.predict(X_test)
y_prob = svm_model.predict_proba(X_test)


# #### Model Evaluation: SVM

# In[12]:


#accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[13]:


#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test, y_pred, average='macro')
print(f"Precision: {precision}")


# In[14]:


#Classification Report
from sklearn.metrics import classification_report

# Define a mapping for class names
class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

target_names = [class_mapping[label] for label in sorted(class_mapping.keys())]

print("\nClassification Report after Hyperparameter Tuning:\n")
print(classification_report(y_test, y_pred, target_names=target_names))


# In[15]:


from sklearn.metrics import recall_score
print("\nRecall (Macro):", recall_score(y_test, y_pred, average='macro'))


# In[16]:


from sklearn.metrics import f1_score
print("\nF1-Score (Macro):", f1_score(y_test, y_pred, average='macro'))


# In[17]:


from sklearn.metrics import matthews_corrcoef
print("\nMatthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))


# In[18]:


from sklearn.metrics import jaccard_score
print("\nJaccard Score (Macro):", jaccard_score(y_test, y_pred, average='macro'))


# In[19]:


from sklearn.metrics import log_loss

#probabilities for all classes
y_prob = svm_model.predict_proba(X_test)

#Passing the true labels explicitly
unique_classes = svm_model.classes_
print("Log Loss:", log_loss(y_test, y_prob, labels=unique_classes))

#y_prob = logistic_model.predict_proba(X_test)[:, 1]


# In[20]:


from sklearn.metrics import ConfusionMatrixDisplay
#Confusion matrix plot
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# In[21]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

#Define a mapping for class names
class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

#Ensuring classes are correctly labeled and binarized
n_classes = len(svm_model.classes_)
y_test_bin = label_binarize(y_test, classes=svm_model.classes_)

#Ensuring probabilities are available for all classes
y_prob = svm_model.predict_proba(X_test)

#Plotting the ROC Curve for each class
plt.figure(figsize=(8, 3))
for i, class_label in enumerate(svm_model.classes_):
    #Compute FPR and TPR
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i]) 
    roc_auc = auc(fpr, tpr)
    #Use custom class name in the legend
    class_name = class_mapping[class_label]
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

#Plotting the diagonal line for random guessing
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")

#Customize the plot
plt.title("ROC Curve for Multiclass Classification (OvR)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# The ROC curve demonstrates the model's ability to distinguish between classes (Ground, Parking, Rooftop).The Area Under the Curve (AUC) values indicate excellent performance:
# Ground: 0.99,
# Parking: 0.99,
# Rooftop: 0.98

# In[22]:


from sklearn.metrics import roc_auc_score

#Getting the probabilities for all classes
y_prob = svm_model.predict_proba(X_test)  

#Use roc_auc_score with 'ovr' or 'ovo' for multiclass
print("ROC-AUC Score (OVR):", roc_auc_score(y_test, y_prob, multi_class='ovr'))
print("ROC-AUC Score (OVO):", roc_auc_score(y_test, y_prob, multi_class='ovo'))


# In[23]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
#Define a mapping for class names
class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

#Binarize the true labels
n_classes = len(np.unique(y_test))  
y_test_bin = label_binarize(y_test, classes=np.unique(y_test)) 

#Confirm the shape of y_prob
y_prob = svm_model.predict_proba(X_test) 

#Plotting the Precision-Recall Curve for each class
plt.figure(figsize=(8, 3))
for i, class_label in enumerate(np.unique(y_test)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    class_name = class_mapping[class_label] 
    plt.plot(recall, precision, label=f"{class_name}")
    
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.show()


# The precision-recall curves for each class also reflect high precision and recall values, signifying minimal false positives and false negatives.

# In[24]:


from sklearn.metrics import mean_absolute_error

#Assuming y_test and y_pred are available
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# In[25]:


from sklearn.metrics import cohen_kappa_score

kappa_score = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa Score:", kappa_score)


# In[26]:


from sklearn.metrics import hinge_loss
decision_scores = svm_model.decision_function(X_test)
hinge_loss_value = hinge_loss(y_test, decision_scores)
print("\nHinge Loss:", hinge_loss_value)


# Key Insights:
# 1. Hyperparameter tuning greatly improved SVM's accuracy from 86.97% to 93.58%, thus showing the real capability of the model with proper tuning of parameters.
# 2. Hyperparameter tuning effectively optimized the decision boundaries leading to better generalization and higher classification performance.
# 3. Hence, the tuned SVM outperforms all other models in overall accuracy and reliability; thus, it is the best candidate for deployment.
# 4. However, targeted enhancements in classes such as Parking and Rooftop still have room for improvement in order to reduce false positives and false negatives.
# 5. Metrics like MCC (0.90) and Jaccard Score (0.88) highlight the SVM's strong correlation and overlap between predictions and true labels, indicating consistent performance across all classes.
