#!/usr/bin/env python
# coding: utf-8

# # Objective:
# 
# Predict the quality of wine (on a scale of 1-10) using the input variables:
# 
# 1. Fixed Acidity value
# 2. Volatile Acidity value
# 3. Citric Acid Content
# 4. Residual Sugar level
# 5. Chlorides level
# 6. Free Sulfur Dioxide
# 7. Total Sulfur Dioxide
# 8. Density
# 9. pH value
# 10. Sulphates
# 11. Alcohol content
# 
# **It is a multi-class classification problem**

# # Loads the necessary libraries and data

# In[ ]:


import pandas as pd
from sklearn.datasets import fetch_openml
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import warnings # not a necessary library. Importing it to suppress warnings so that the output looks clean
warnings.filterwarnings("ignore")


# ### Fetch red wine quality data from fetch_openml dataset

# In[ ]:


wine_data = fetch_openml(name='wine-quality-red')


# In[6]:


wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target_quality'] = wine_data.target
wine_df.head()


# ### Checking the data type of each predictor and target

# In[7]:


wine_df.dtypes


# Change the data type of `target_quality` to integer

# In[ ]:


wine_df['target_quality'] = wine_df['target_quality'].astype('int64')


# In[9]:


wine_df.describe()


# # Exploratory Data Analysis and Feature Derivation

# ### Checking the number of NULL values in each column

# In[10]:


wine_df.isnull().sum()


# ### **Histogram** - count plot of the target variable
# It is used in univariate analysis, they give a rough sense of density of underlying distribution of data more precisely probability distribution of data. The plotting of the histogram depends upon ‘bins’ 

# In[11]:


wine_df['target_quality'].unique()


# In[12]:



sns.countplot(x='target_quality', data=wine_df)


# * The above distribution shows the range for response variable (target_quality) is between 3 to 8.
# 
# 
# 
# 

# ### Corelation between features/variables:

# In[13]:


correlation = wine_df.corr()
# sns.heatmap(uniform_data)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation, cmap=cmap, vmin=0, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)


# * The squares with positive values show direct co-relationships between features. The higher the values, the stronger these relationships are — they’ll be more reddish. That means, if one feature increases, the other one also tends to increase, and vice-versa.
# 
# * The squares that have negative values show an inverse co-relationship. The more negative these values get, the more inversely proportional they are, and they’ll be more blue. This means that if the value of one feature is higher, the value of the other one gets lower.
# 
# * Finally, squares close to zero indicate almost no co-dependency between those sets of features.

# ### Categorizing the quality into bad, average and good
# * Create a new discreet, categorical response variable/feature ('rating') from existing 'quality' variable.
# i.e.
#       bad: 1-4
#       average: 5-6
#       good: 7-10

# In[ ]:


reviews = []
for i in wine_df['target_quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
wine_df['reviews'] = reviews


# # Training the ML models and evaluating them - Classification

# ## Model evaluation metrics:
# 
# 
# ### Confusion Matrix
# The confusion matrix for a multi-class classification problem can help you determine mistake patterns.
# 
# For a Binary Classifier,
# ![alt text](https://cdn-images-1.medium.com/max/2000/1*mrBc8OW0NfXv43fpSnMa9g.png)
# 
# **True Positive** : A true positive is an outcome where the model correctly predicts the positive class. 
# 
# **True Negative**: A true negative is an outcome where the model correctly predicts the negative class.
# 
# **False Positive** : An outcome in which the model mistakenly predicted the positive class. 
# 
# **False Negative**: An outcome in which the model mistakenly predicted the negative class
# 
# ### Accuracy
# Accuracy is the fraction of predictions our model got right.
# 
# ![alt text](https://cdn-images-1.medium.com/max/2000/1*i2kn5c_o0ZWu9ncUWCu8bg.png)
# 
# ### Precision
# Out of all the classes, how much we predicted correctly. Precision should be as high as possible.
# 
# 
# ### Recall
# Out of all the positive classes, how much we predicted correctly. It is also called sensitivity or true positive rate (TPR).
# 
# ### F1 score
# It is often convenient to combine precision and recall into a single metric called the F1 score, in particular, if you need a simple way to compare two classifiers. The F1 score is the harmonic mean of precision and recall.
# 
# ![alt text](https://cdn-images-1.medium.com/max/2000/1*UJxVqLnbSj42eRhasKeLOA.png)
# 
# ![alt text](https://s3.amazonaws.com/stackabuse/media/understanding-roc-curves-python-2.png)
# 
# ### Receiver Operator Curve(ROC) & Area Under the Curve(AUC)
# ROC curve is an important classification evaluation metric. It tells us how good the model is accurately predicted. The ROC curve shows the sensitivity of the classifier by plotting the rate of true positives to the rate of false positives. If the classifier is outstanding, the true positive rate will increase, and the area under the curve will be close to 1. If the classifier is similar to random guessing, the true positive rate will increase linearly with the false positive rate. The better the AUC measure, the better the model. Keep in mind that the AUC-ROC curve is plotted for each class outcome separately
# ![alt text](https://cdn-images-1.medium.com/max/2000/1*PqbUAfo5VKom_UriSQsCKQ.png)

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ## Train-Test split (Holdout method)
# Split the data into 2 sets - 80% for training and 20% for testing
# The model will be trained on 80% of the data and tested on the unseen data (remaining 20%). The quality of the model is decided on the basis of performance on unseen data

# In[ ]:


trainX = wine_df.drop(['target_quality', 'reviews'] , axis = 1)
trainy = wine_df['reviews']
X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size = 0.2, random_state = 42)


# ## Standardize features by removing the mean and scaling to unit variance
# 
# Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

# In[ ]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
score = {}


# ## Logistic Regression Model

# In[19]:


lr = LogisticRegression() # declaring the ML method which is Logistic Regression
lr.fit(X_train, y_train) # training the model
predicted_lr = lr.predict(X_test) # predicting the values on unseen data
print(classification_report(y_test, predicted_lr))

lr_conf_matrix = confusion_matrix(y_test, predicted_lr)
lr_acc_score = accuracy_score(y_test, predicted_lr)

print(lr_conf_matrix)
print(lr_acc_score*100)

score.update({'logistic_regressor': lr_acc_score*100})


# ## Decision Tree Model

# In[20]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
predicted_dt = dt.predict(X_test)
print(classification_report(y_test, predicted_dt))

dt_conf_matrix = confusion_matrix(y_test, predicted_dt)
dt_acc_score = accuracy_score(y_test, predicted_dt)
print(dt_conf_matrix)
print(dt_acc_score*100)
score.update({'DecisionTreeClassifier': dt_acc_score*100})


# There could be a possibility that the modela has **OVERFITTED**. There are methods to counter overfitting - the K-Fold Cross validation method. 

# # Training the ML models and evaluating them - Regression

# ## Model evaluation metrics
# 
# **RMSE (Root Mean Square Error)**
# 
# It represents the sample standard deviation of the differences between predicted values and observed values (called residuals). Mathematically, it is calculated using this formula:
# 
# ![alt text](https://cdn-images-1.medium.com/max/2000/1*pQR9id8CtnsdKljm8KODuw.png)
# 
# **MAE**
# 
# MAE is the average of the absolute difference between the predicted values and observed value. The MAE is a linear score which means that all the individual differences are weighted equally in the average.
# 
# ![alt text](https://cdn-images-1.medium.com/max/2000/1*iLabSjpdwd1TaZyKdDKYBA.png)
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


trainX = wine_df.drop(['target_quality', 'reviews'] , axis = 1)
trainy = wine_df['target_quality']
X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size = 0.2, random_state = 42)


# In[ ]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
score = {}


# ## Linear Regression Model

# In[24]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[25]:


y_prediction = regressor.predict(X_test)
# Let's see the first few outputs of the regression model
print(y_prediction[:5]) 


# In[26]:


# Calculate the RMSE value
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)


# # Further readings
# 
# *  Support Vector Machine ML model
# * **Ensemble methods**
# * Bagging and Boosting
# * RandomForest ML model
# * XGBoost ML model
# * Feature selection
# * Regression - MSE, MAE, R Squared
# * Time series modeling
# * Clustering - K Means. PCA
# * Neural Networks
# * Overfitting
# * K-Fold Cross Validation to reduce overfitting
# * Feature engineering - more methods
# * Parameter tuning
# 
# 

# In[ ]:




