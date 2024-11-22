#!/usr/bin/env python
# coding: utf-8

# **GBM, XGBoost algorithms getting considered in this notebook**

# In[21]:


#importing all the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,matthews_corrcoef,jaccard_score, accuracy_score,confusion_matrix,ConfusionMatrixDisplay,roc_auc_score,roc_curve,auc,precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import  GridSearchCV,RandomizedSearchCV
import xgboost as xgb


# In[5]:


#loading the transformed Data Set
solar_data = pd.read_csv(r"solar_data_transformed18.csv")
print(solar_data.columns)


# In[6]:


X = solar_data.drop(columns=['ID','InstallType'])  # Features
y = solar_data['InstallType']  # Target


# #### Gradient Boosting Machine (GBM) Model Implementation

# In[7]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the GBM model
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)

# Make predictions
y_pred = gbm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)



# In[9]:


print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nGBM Classification Report:\n", classification_rep)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')

y_prob = gbm_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("ROC-AUC Score:", roc_auc)


# #### Fine tuning

# In[10]:


# Define parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

# Grid Search
gbm = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# In[11]:


# Predict on the test set
y_pred_gbm_tuned = best_model.predict(X_test)

# Calculate accuracy
gbm_tuned_accuracy = accuracy_score(y_test, y_pred_gbm_tuned)

# Classification report for detailed metrics
gbm_tuned_report = classification_report(y_test, y_pred_gbm_tuned)

print("Tuned GBM Accuracy:", gbm_tuned_accuracy)
print("Tuned GBM Classification Report:\n", gbm_tuned_report)

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_gbm_tuned))

conf_matrix=confusion_matrix(y_test, y_pred_gbm_tuned)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')

y_prob = best_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("ROC-AUC Score:", roc_auc)


# In[12]:


# Ensure binary format for true labels
n_classes = len(best_model.classes_)
y_test_bin = label_binarize(y_test, classes=best_model.classes_)

# Get predicted probabilities
y_prob = best_model.predict_proba(X_test)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Plot Precision-Recall Curves
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    axes[0].plot(recall, precision, label=f"Class {best_model.classes_[i]}")

axes[0].set_title("Precision-Recall Curve (GBM)", fontsize=12)
axes[0].set_xlabel("Recall", fontsize=10)
axes[0].set_ylabel("Precision", fontsize=10)
axes[0].legend(loc="lower left", fontsize='small')
axes[0].grid(alpha=0.3)

# Plot ROC Curves
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f"Class {best_model.classes_[i]} (AUC = {roc_auc:.2f})")

axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
axes[1].set_title("ROC Curve (GBM)", fontsize=12)
axes[1].set_xlabel("False Positive Rate", fontsize=10)
axes[1].set_ylabel("True Positive Rate", fontsize=10)
axes[1].legend(loc="lower right", fontsize='small')
axes[1].grid(alpha=0.3)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# In[17]:


#Use roc_auc_score with 'ovr' or 'ovo' for multiclass
print("ROC-AUC Score (OVR):", roc_auc_score(y_test, y_prob, multi_class='ovr'))
print("ROC-AUC Score (OVO):", roc_auc_score(y_test, y_prob, multi_class='ovo'))
from sklearn.metrics import cohen_kappa_score
kappa_score = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa Score:", kappa_score)
print("Jaccard Score (Macro):", jaccard_score(y_test, y_pred, average='macro'))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))


# #### GBM Insights

# The tuned GBM outperforms the original model slightly. That means the hyperparameter tuning really helped this model to perform better. This increased the accuracy from 92.5% to 96.66%, showing that fine-tuning reduced minor classification errors. Precision, recall, and F1-scores are all above 95 for the general performance, meaning high consistency in the performance across the classes. Tuning these metrics does not break, and there is a balance held across the dataset. The tuned GBM outperforms the original model slightly. That means the hyperparameter tuning really helped this model to perform better.

# #### XGBoost implementation

# In[18]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
classification_rep_xgb = classification_report(y_test, y_pred_xgb)

# Print evaluation results
print("XGBoost Accuracy:", accuracy_xgb)
print("\nConfusion Matrix:\n", conf_matrix_xgb)
print("\nXGBoost Classification Report:\n", classification_rep_xgb)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("ROC-AUC Score:", roc_auc)

# Display the confusion matrix
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_xgb)
disp_xgb.plot(cmap='Blues')  


# #### Fine tuning

# In[22]:


# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5],
}

# Set up the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings to sample
    scoring='accuracy',  # Optimize for accuracy
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Train the model with the best parameters
best_xgb_model = random_search.best_estimator_

# Make predictions
y_pred_xgb = best_xgb_model.predict(X_test)
y_prob_xgb = best_xgb_model.predict_proba(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
classification_rep_xgb = classification_report(y_test, y_pred_xgb)
roc_auc = roc_auc_score(y_test, y_prob_xgb, multi_class='ovr')

# Print evaluation results
print("XGBoost fine tuned Accuracy:", accuracy_xgb)
print("\nConfusion Matrix:\n", conf_matrix_xgb)
print("\nXGBoost Classification Report:\n", classification_rep_xgb)
print("ROC-AUC Score:", roc_auc)

# Display the confusion matrix
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_xgb)
disp_xgb.plot(cmap='Blues')


# In[23]:


# Ensure binary format for true labels
n_classes = len(xgb_model.classes_)
y_test_bin = label_binarize(y_test, classes=xgb_model.classes_)

# Get predicted probabilities
y_prob = xgb_model.predict_proba(X_test)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Plot Precision-Recall Curves
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    axes[0].plot(recall, precision, label=f"Class {xgb_model.classes_[i]}")

axes[0].set_title("Precision-Recall Curve (XGBoost)", fontsize=12)
axes[0].set_xlabel("Recall", fontsize=10)
axes[0].set_ylabel("Precision", fontsize=10)
axes[0].legend(loc="lower left", fontsize='small')
axes[0].grid(alpha=0.3)

# Plot ROC Curves
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f"Class {xgb_model.classes_[i]} (AUC = {roc_auc:.2f})")

axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
axes[1].set_title("ROC Curve (XGBoost)", fontsize=12)
axes[1].set_xlabel("False Positive Rate", fontsize=10)
axes[1].set_ylabel("True Positive Rate", fontsize=10)
axes[1].legend(loc="lower right", fontsize='small')
axes[1].grid(alpha=0.3)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()



# In[24]:


#Use roc_auc_score with 'ovr' or 'ovo' for multiclass
print("ROC-AUC Score (OVR):", roc_auc_score(y_test, y_prob, multi_class='ovr'))
print("ROC-AUC Score (OVO):", roc_auc_score(y_test, y_prob, multi_class='ovo'))
#print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred_dt))
from sklearn.metrics import cohen_kappa_score
kappa_score = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa Score:", kappa_score)
print("Jaccard Score (Macro):", jaccard_score(y_test, y_pred, average='macro'))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))


# #### XGBoost insights

# The XGBoost model, before tuning, gave an accuracy of 96.48% with an excellent ROC-AUC score of 0.996, which signifies very good classification performance. The classification report showed very high precision and recall for all the classes, the majority class 0 having a recall of 0.97 and 0.95 for class 1 and class 2, respectively. From the confusion matrix, it was found that there were very few misclassifications. Fine-tuning has resulted in accuracy, which has remained at 96.02%, while the confusion matrix has changed only a little to reflect the tiny increase in the misclassifications of class 0. Overall, the performance is excellent, with the ROC-AUC score decreased very slightly at 0.995, indicating that after tuning, there is only a marginal effect.

# In[ ]:




