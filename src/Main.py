# importamos todas las librerias
import joblib
import numpy as np
import pandas as pd
import re
import warnings

from catboost import CatBoostClassifier

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# cargamos los datos

data = pd.read_csv('data/ChurnModeling.csv')
print(data.head()) # vemos las variables
data.pop('RowNumber') # removemos la columna RowNumber

# Separación train y test:
features = ['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
       'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
       'EstimatedSalary']
X = data.loc[:, features]
y = data.loc[:, ['Exited']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Variables que excluir
exclude_columns = ["CustomerId", "Surname"]
# Variables categoricas
cat_var = ['Geography', 'Gender']
# Variables numericas
num_var = ['CreditScore', 'Age',
       'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
       'EstimatedSalary']

# Creamos el pipeline del preprocesado de datos:
cat_pipeline = Pipeline(
    [("Impute_Mode", SimpleImputer(strategy = "most_frequent")),
     ("OHEncoder", OneHotEncoder())])



num_pipeline = Pipeline(
    [("Impute_Mean", SimpleImputer(strategy = "mean")),
     ("SScaler", StandardScaler())])


preprocessing = ColumnTransformer(
    [("Process_Categorical", cat_pipeline, cat_var),
     ("Impute_Numeric", num_pipeline, num_var),
     ("Exclude", "drop", exclude_columns)
    ], remainder = "passthrough")

# PCA de los datos

pca = PCA(n_components=9)

catb_pipeline = Pipeline(
    [("Preprocesado", preprocessing),
     ("pca", pca),
     ("Modelo", CatBoostClassifier())
    ])

# Grid search:

pipe_catb_param = {
    'Modelo__n_estimators': [330,345,360,365],
    'Modelo__scale_pos_weight':[8],
    'Modelo__max_depth': [2,3,4],
    'Modelo__learning_rate': [0.88, 0.9, 0.92]}

cv = 8

gs_catb = GridSearchCV(catb_pipeline,
                        pipe_catb_param,
                        cv=cv,
                        scoring="recall",
                        verbose=1,
                        n_jobs=-1)

gs_catb.fit(X_train, y_train)

# Modelo optimizado:

best_catb = gs_catb.best_estimator_

y_pred_catb = best_catb.predict(X_test)

for metric,evaluator in zip(["Precision","Recall","Accuracy","f1_score"],[precision_score, recall_score, \
                                                                              accuracy_score, f1_score]):
        valor = evaluator(y_test,y_pred_catb)
        print("%s: %.2f" %(metric, valor))

# Guardamos el modelo
joblib.dump(best_catb, "churn_prediction_model.pkl")