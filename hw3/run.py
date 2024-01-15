import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor


parser = argparse.ArgumentParser()
parser.add_argument('--train-data', type=str, default='./my_train.csv', help='path to training data')
parser.add_argument('--dev-data', type=str, default='./my_dev.csv', help='path to development data')
parser.add_argument('--test-data', type=str, default='./test.csv', help='path to test data')
parser.add_argument('--output', type=str, default='./test_submission.csv', help='path to output file')
parser.add_argument('--run-test', action='store_true', default=False, help='whether to run on test data')
parser.add_argument('-m', '--model', type=int, default=1, help='model to use: [0: Linear Regression, 1: Ridge Regression]')
parser.add_argument('-p', '--preprocessor', type=int, default=2, help='preprocessor to use: [0: Naive Binarization, 1: Smart Binarization, 2: Smart Binarization + Non-Linear Transformations]')
parser.add_argument('--verbose', action='store_true', default=False, help='whether to print verbose output')

if __name__ == '__main__':
    args = parser.parse_args()

    # load the training data
    train_data = pd.read_csv(args.train_data).drop("Id", axis=1)
    for col in train_data.columns:  # convert all integer columns to float
        if pd.api.types.is_integer_dtype(train_data[col]):
            train_data[col] = train_data[col].astype(float)
    train_data_types = train_data.dtypes.apply(lambda x: x.name).to_dict()
    test_data_types = train_data_types.copy()
    test_data_types.pop("SalePrice")
    # load the development data
    dev_data = pd.read_csv(args.dev_data, dtype=train_data_types).drop("Id", axis=1)
    # load the test data
    test_data = pd.read_csv(args.test_data, dtype=test_data_types)


    print("Train Data Shape:", train_data.shape)
    print("Dev Data Shape:", dev_data.shape)
    print("Test Data Shape:", test_data.shape)


    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_dev = dev_data.iloc[:, :-1]
    y_dev = dev_data.iloc[:, -1]
    X_test = test_data.iloc[:, 1:]


    # identify categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    num_cols = X_train.select_dtypes(exclude=['object']).columns

    print("Number of categorical columns:", len(cat_cols))
    print("Number of numerical columns:", len(num_cols))


    # define transformers
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    # naive preprocessor: NAIVE BINARIZATION
    naive_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[('to_str', FunctionTransformer(lambda X: X.astype(str))), 
            ('one_hot', OneHotEncoder(handle_unknown='ignore'))]), X_train.columns),
        ]
    )

    # smart preprocessor: SMART BINARIZATION
    smart_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ]
    )

    # smart preprocessor: SMART BINARIZATION + NON-LINEAR TRANSFORMATIONS
    non_linear_transform_columns = ['LotArea', 'LotFrontage', 'TotalBsmtSF']
    non_linear_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly', PolynomialFeatures(degree=2, include_bias=True)),
        ('scaler', StandardScaler())
    ])
    non_linear_smart_preprocessor = ColumnTransformer(transformers=[
        ('non-linear-transformer', non_linear_transformer, non_linear_transform_columns),
        ('num', numerical_transformer, [col for col in num_cols if col not in non_linear_transform_columns]),
        ('cat', categorical_transformer, cat_cols)
        ]
    )

    if args.preprocessor == 0:
        preprocessor = naive_preprocessor
    elif args.preprocessor == 1:
        preprocessor = smart_preprocessor
    else:
        preprocessor = non_linear_smart_preprocessor
    

    model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression() if args.model == 0 else Ridge(alpha=0.01)),
                ])

    print("\n[INFO] Using", "Smart Binarization" if args.preprocessor == 1 else "Naive Binarization")
    print("[INFO] Using", "Linear Regression" if args.model == 0 else "Ridge Regression")

    # train the model
    model.fit(X_train, np.log(y_train))     # Using log of y_train for RMSLE

    if args.verbose:
        # print the intercept
        print("[INFO] Intercept:", model.named_steps['regressor'].intercept_)
        print("[INFO] Number of Coeffecients:", len(model.named_steps['regressor'].coef_))

        # get top 10 most positive features
        print("\n[INFO] Top 10 most positive features:")
        print("--------------------------------------")
        coef = model.named_steps['regressor'].coef_
        top10_pos = np.argsort(coef)[-10:]
        if args.preprocessor == 0:
            features = preprocessor.named_transformers_['cat'].named_steps['one_hot'].get_feature_names_out()
        else:
            features = np.array([feature.split('__')[-1] for feature in preprocessor.get_feature_names_out()])
        top_10_pos_features = sorted(zip(features[top10_pos], coef[top10_pos]), key=lambda x: x[1], reverse=True)
        print("Feature{:<20}Coefficient".format(""))
        print("--------------------------------------")
        print("\n".join([f"{feature:<30}{coef:.4f}" for feature, coef in top_10_pos_features]))

        # get top 10 most negative features
        print("\n[INFO] Top 10 most negative features:")
        print("--------------------------------------")
        top10_neg = np.argsort(coef)[:10]
        if args.preprocessor == 0:
            features = preprocessor.named_transformers_['cat'].named_steps['one_hot'].get_feature_names_out()
        else:
            features = np.array([feature.split('__')[-1] for feature in preprocessor.get_feature_names_out()])
        top_10_neg_features = sorted(zip(features[top10_neg], coef[top10_neg]), key=lambda x: x[1])
        print("Feature{:<20}Coefficient".format(""))
        print("--------------------------------------")
        print("\n".join([f"{feature:<29}{coef:.4f}" for feature, coef in top_10_neg_features]))
        print()

    y_pred_train = model.predict(X_train)   # predict on the training set
    y_pred_dev = model.predict(X_dev)       # predict on the development set

    
    # calculate RMSLE
    train_rmsle = np.sqrt(mean_squared_error(np.log(y_train), y_pred_train))
    dev_rmsle = np.sqrt(mean_squared_error(np.log(y_dev), y_pred_dev))
    print("Train RMSLE:", train_rmsle)
    print("Dev RMSLE:", dev_rmsle)

    if args.run_test:
        # predict on the test set
        y_pred_test = model.predict(X_test)
        # convert to actual prices
        y_pred_test = np.exp(y_pred_test)
        # create a DF with ID and SalePrice columns
        y_pred_test = pd.DataFrame({'Id': test_data["Id"], 'SalePrice': y_pred_test})
        # output with header
        y_pred_test.to_csv(args.output, index=False)
        print("Test predictions saved to", args.output)