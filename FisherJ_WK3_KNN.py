#set working directory
import os
os.chdir('C:/Users/unkno/Desktop/MS Data Science/Class 9 - ANA680/Week 3/src')

#import libraries
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
import pickle as pkl

#fetch data
wine_quality = fetch_ucirepo(id=186)

#separate features and target 
X = wine_quality.data.features 
y = wine_quality.data.targets 

#flatten y into a 1D array
y = np.ravel(y)

#check for NaN values
print('Number of NaN values in X: ', X.isnull().sum())

#check for NaN values
print('Number of NaN values in X: ', np.isnan(X).sum())

#print minimum for each feature
print('Minimum values for each feature: ', X.min())

#print maximum for each feature
print('Maximum values for each feature: ', X.max())

#split data into training and testing sets (testing = 25% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

## RFE FEATURE SELECTION
#initialize a Logistic Regression model to use for feature selection
model = LogisticRegression()

#count number of features
n_features = X_train.shape[1]
print('Number of Features: ', n_features)

#print features
print('Features: ', X.columns)

# Create the RFE model and select the top features
rfe = RFE(estimator=model, n_features_to_select=7)  # Adjust n_features_to_select to your needs
rfe = rfe.fit(X_train, y_train)

# Print selected features
print('Selected Features: ', rfe.support_)
print('Feature Ranking: ', rfe.ranking_)

# Transform the dataset to contain only the selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

#print the features selected
print('Features Selected: ', X.columns[rfe.support_])

## removed residual_sugar, chlorides, free_sulfur_dioxide, and total_sulfur_dioxide 
## simplified model (and k=4) with only 7 of the 11 features is higher (55.75% vs 49.66%)

# Reinitialize and train KNN model on the 7 selected features
model_rfe = KNeighborsClassifier(n_neighbors=4)
model_rfe.fit(X_train_rfe, y_train)

# Predict the test set
y_pred_rfe = model_rfe.predict(X_test_rfe)

# Calculate accuracy
accuracy_rfe = accuracy(y_test, y_pred_rfe)
print('Accuracy after RFE: ', accuracy_rfe)

# Confusion matrix
conf_matrix_rfe = confusion_matrix(y_test, y_pred_rfe)
print('Confusion Matrix after RFE: ')
print(conf_matrix_rfe)

#print the features selected
print('Features Selected: ', X.columns[rfe.support_])

# Save the model
pkl.dump(model_rfe, open('knn_model.pkl', 'wb'))