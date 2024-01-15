import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer

# Custom scorer for RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log(y_true + 1), np.log(y_pred + 1)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = np.log(train_data.SalePrice)
X = train_data.drop(['SalePrice'], axis=1)

categorical_cols = [cname for cname in X.columns if
                    X[cname].dtype == "object"]

numerical_cols = [cname for cname in X.columns if 
                  X[cname].dtype in ['int64', 'float64']]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()) # Normalization step
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

non_linear_transform_columns = []
non_linear_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=True)),
    ('scaler', StandardScaler())
])
non_linear_smart_preprocessor = ColumnTransformer(transformers=[
    ('non-linear-transformer', non_linear_transformer, non_linear_transform_columns),
    ('num', numerical_transformer, [col for col in numerical_cols if col not in non_linear_transform_columns]),
    ('cat', categorical_transformer, categorical_cols)
    ]
)
preprocessor = non_linear_smart_preprocessor

# Define the model
model = XGBRegressor()

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Hyperparameters to tune
param_grid = {
    'model__n_estimators': [500],
    'model__learning_rate': [0.1],
    'model__max_depth': [3]
}

# Setup the grid search
grid_search = GridSearchCV(clf, param_grid, cv=5, return_train_score=True)

# Fit the grid search to the data
grid_search.fit(X, y)

model = grid_search.best_estimator_
y_pred_test = model.predict(test_data)
y_pred_test = np.exp(y_pred_test)  # Inverse of logarithm
# create a DF with ID and SalePrice columns
y_pred_test = pd.DataFrame({'Id': test_data["Id"], 'SalePrice': y_pred_test})
# output with header
y_pred_test.to_csv('test_submission.csv', index=False)
print("Test predictions saved to", 'test_submission.csv')


