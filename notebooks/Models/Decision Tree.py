#!/usr/bin/env python
# coding: utf-8

# **Decision Tree**

# In[1]:


#importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score,confusion_matrix,ConfusionMatrixDisplay,roc_auc_score,roc_curve,auc,precision_recall_curve,mean_absolute_error,log_loss,jaccard_score,matthews_corrcoef,cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import  GridSearchCV


# In[3]:


#loading the transformed Data Set
solar_data = pd.read_csv(r"C:\Users\bathl\OneDrive\Documents\SJSU\Machine_Learning\Project\solar_data_transformed18.csv")


# In[5]:


# Exclude the target and ID column from the PCA analysis
X = solar_data.drop(['InstallType','ID'], axis=1)

# Prepare data for classification
X_train, X_test, y_train, y_test = train_test_split(X, solar_data['InstallType'], test_size=0.3, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_test, y_pred_dt)

#target column values' mapping
class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

target_names = [class_mapping[label] for label in sorted(class_mapping.keys())]

dt_report = classification_report(y_test, y_pred_dt,target_names=target_names)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Classification Report:\n", dt_report)


# In[7]:


#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

disp.plot(cmap='Blues')
# Add labels to the plot
class_names = ['Ground', 'Parking', 'Rooftop']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Decision Tree Model')
plt.show()


# In[8]:


# Decision Tree Tuning
dt_param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'] 
}

dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
dt_grid_search.fit(X_train, y_train)


# In[9]:


# Best parameters and performance
best_dt = dt_grid_search.best_estimator_
y_pred_dt_tuned = best_dt.predict(X_test)
dt_tuned_accuracy = accuracy_score(y_test, y_pred_dt_tuned)

#target column values' mapping
class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

target_names = [class_mapping[label] for label in sorted(class_mapping.keys())]

dt_tuned_report = classification_report(y_test, y_pred_dt_tuned,target_names=target_names)

print("Tuned Decision Tree Accuracy:", dt_tuned_accuracy)
print("Tuned Decision Tree Classification Report:\n", dt_tuned_report)


# In[11]:


conf_matrix = confusion_matrix(y_test, y_pred_dt_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')

# Add labels to the plot
class_names = ['Ground', 'Parking', 'Rooftop']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Decision Tree Model')
plt.show()


# **DecisionTree Confusion Matrix Insights**

# The performance metrics are a bit different for the Decision Tree and its tuned variant; the original Decision Tree is slightly more accurate than the tuned model.
# 
# 1. The accuracy differs from 91.04% to 90.61% indicating Decision tree before fine tuning is slightlty better.
# 2. Precision, recall, and F1-scores are consistent around 0.91 across both models, indicating that both models provide comparable reliability for classification tasks.
# 3. However, its performance lags behind the Random Forest, suggesting that ensemble methods remain more effective for this dataset.

# In[14]:


#precision
print("\nPrecision (Macro):", precision_score(y_test, y_pred_dt, average='macro'))


# In[15]:


#recall
print("\nRecall (Macro):", recall_score(y_test, y_pred_dt, average='macro'))


# In[16]:


#F1-score
print("\nF1-Score (Macro):", f1_score(y_test, y_pred_dt, average='macro'))


# In[23]:


#Define a mapping for class names
class_mapping = {
    0: "Ground",
    1: "Parking",
    2: "Rooftop"
}

# Ensure classes are correctly labeled and binarized
n_classes = len(best_dt.classes_)  # Number of classes
y_test_bin = label_binarize(y_test, classes=best_dt.classes_)  # Binarize y_test

# Ensure probabilities are available for all classes
y_prob = best_dt.predict_proba(X_test)  # Predicted probabilities

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 

# Plot Precision-Recall Curves
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    pr_auc = auc(recall, precision)
    class_name = class_mapping[best_dt.classes_[i]] 
    axes[0].plot(recall, precision, label=f"{class_name}(AUC = {pr_auc:.2f})")

axes[0].set_title("Precision-Recall Curve (Decision Tree)", fontsize=12)
axes[0].set_xlabel("Recall", fontsize=10)
axes[0].set_ylabel("Precision", fontsize=10)
axes[0].legend(loc="lower left", fontsize='small')
axes[0].grid(alpha=0.3)


# Plotting the ROC Curve for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])  # Compute FPR and TPR
    roc_auc = auc(fpr, tpr)  # Compute AUC
    class_name = class_mapping[best_dt.classes_[i]]
    plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")

# Plot the diagonal line for random guessing
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")

# Customize the plot
plt.title("ROC Curve for Multiclass Classification (OvR)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# Precision-Recall Curve:
# 
# 1. All three classes experience a decline in precision as recall approaches 1.0, indicating a higher proportion of false positives at these thresholds.
# 2. The Rooftop class shows the sharpest drop, suggesting the model struggles more with distinguishing this class compared to the others.
# 3. Lower AUC values for Parking and Rooftop suggest these classes might be harder to predict due to class overlap or fewer training examples.
# 4. Decision Trees are prone to overfitting, which might explain the less smooth curves and slightly lower AUC values compared to ensemble models like Random Forests.
# 
# ROC-AUC curve:
# 
# 1. The AUC values are close across all three classes, indicating relatively balanced discrimination capability.
# 2. The slightly lower AUC for Rooftop suggests that this class is more challenging to differentiate.

# In[26]:


#Use roc_auc_score with 'ovr' or 'ovo' for multiclass
print("ROC-AUC Score (OVR):", roc_auc_score(y_test, y_prob, multi_class='ovr'))
print("ROC-AUC Score (OVO):", roc_auc_score(y_test, y_prob, multi_class='ovo'))


# In[28]:


print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred_dt))


# In[30]:


print("Jaccard Score (Macro):", jaccard_score(y_test, y_pred_dt, average='macro'))


# In[34]:


mae = mean_absolute_error(y_test, y_pred_dt)
print(f"Absolute Loss (Mean Absolute Error): {mae:.4f}")


# In[36]:


kappa_score = cohen_kappa_score(y_test, y_pred_dt)
print("Cohen's Kappa Score:", kappa_score)


# In[ ]:





# In[ ]:




