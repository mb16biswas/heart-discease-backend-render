
"""
could not import the model in Server.py  that was trained in jupytor notebook 
by knowing  the best model and best  parameters 
recreating the same model 
 
"""


import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv("heart-disease-problem.csv")
X = df.drop("target" , axis= 1 )
y = df["target"]
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2)
model = RandomForestClassifier( n_estimators = 1200,
                                min_samples_split = 2,
                                min_samples_leaf = 4,
                                max_features = 'auto',
                                max_depth= 100,
                                bootstrap= True)

model.fit(X,y)



pickle.dump(model, open("heart_class.pkl", "wb"))
