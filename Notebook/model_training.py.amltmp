#!/usr/bin/env python
# coding: utf-8

# In[3]:


from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the data

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset=ml_client.data.get("Social_Network_Ads", version="1")
data_asset.path

df=pd.read_csv(data_asset.path)
df.head(5)

# Prepare the X and y variable


X=df.drop('Purchased',axis=1)
y=df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the X variable


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#Set the experiment and Enable autologging**


mlflow.set_experiment('Model Run from Notebook')
mlflow.sklearn.autolog()


# Train the Random Forest model
# Run the experiment with Random Forest classifier

mlflow.start_run(run_name='Random_Forest')
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
mlflow.end_run()


# Train SVM model
# Run another experiment with SVM Classifier


mlflow.start_run(run_name='SVM_Classifier')
model1=SVC(kernel='rbf')
model1.fit(X_train,y_train)
prediction=model.predict(X_test)
mlflow.end_run()

