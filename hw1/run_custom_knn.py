# flake8: noqa
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from knn_classifier import KNNClassifier
from evaluate import evaluate_custom_knn


# get data path
parser = argparse.ArgumentParser()
parser.add_argument('--train-data', type=str, default='./data/income.train.txt.5k')
parser.add_argument('--dev-data', type=str, default='./data/income.dev.txt')
parser.add_argument('--test-data', type=str, default='./data/income.test.blind')
parser.add_argument('--test-output', type=str, default='./data/income.test.predicted')
parser.add_argument('-k', '--k', type=int, default=41, help='number of nearest neighbors')
parser.add_argument('-o', '--order', type=int, default=2, help='order of norm')
parser.add_argument('--min-k', type=int, default=1, help='minimum k for evaluation')
parser.add_argument('--max-k', type=int, default=101, help='maximum k for evaluation')
parser.add_argument('--test', action='store_true')
parser.add_argument('--eval', action='store_true')

args = parser.parse_args()


if __name__ == '__main__':
    # loading data
    column_names = ["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"]
    train_data = pd.read_csv(args.train_data, names=column_names)
    if args.test:
        test_data = pd.read_csv(args.test_data, names=column_names[:-1])
    else:
        test_data = pd.read_csv(args.dev_data, names=column_names)

    # preprocessing data
    num_processor = MinMaxScaler(feature_range=(0, 2))
    cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    num_columns = list(train_data.select_dtypes(include=['int64', 'float64']).columns)
    cat_columns = list(train_data.select_dtypes(include=['object']).columns)

    preprocessor = ColumnTransformer([
            ('num', num_processor, num_columns),
            ('cat', cat_processor, cat_columns)
        ])

    if args.test:
        cat_columns.remove('target')
        X_train = preprocessor.fit_transform(train_data.drop(columns=['target']))
        X_test = preprocessor.transform(test_data)

        label_processor = OneHotEncoder(sparse_output=False)
        y_train = label_processor.fit_transform(train_data['target'].values.reshape(-1, 1))

        knn = KNNClassifier(args.k)
        knn.fit(X_train, y_train)

        predictions = label_processor.inverse_transform(knn.predict(X_test, order=args.order))
        # write predictions to file
        with open(args.test_output, 'w') as f:
            for i, prediction in enumerate(predictions):
                # convert test data line to single line and add prediction, also make sure everything is string
                f.write(', '.join([str(x).strip() for x in test_data.iloc[i]]) + ', ' + prediction[0].strip() + '\n')
        
        # calculate error train predictions
        # import numpy as np
        # train_predictions = label_processor.inverse_transform(knn.predict(X_train, order=args.order))
        # train_accuracy = np.mean(train_predictions == train_data['target'].values.reshape(-1, 1))
        # train_error = 1 - train_accuracy
        # train_positive_rate = np.mean(train_predictions == ' >50K')
        # print("Train Accuracy: {:.3f}".format(train_accuracy))
        # print("Train Error: {:.3f}".format(train_error))
        # print("Train Positive Rate: {:.3f}".format(train_positive_rate))

        exit(0)


    preprocessor.fit(train_data)
    train_data_processed = preprocessor.transform(train_data)
    test_data_processed = preprocessor.transform(test_data)
    feature_dimension_size = train_data_processed.shape[1]

    X_train = train_data_processed[:, :feature_dimension_size - 2]
    X_test = test_data_processed[:, :feature_dimension_size - 2]
    y_train = train_data_processed[:, feature_dimension_size - 2:]
    y_test = test_data_processed[:, feature_dimension_size - 2:]

    knn = KNNClassifier(args.k)
    knn.fit(X_train, y_train)

    if args.eval:
        evaluate_custom_knn(X_train, y_train, X_test, y_test, order=args.order)
    else:
        knn.evaluate(X_test, y_test, min_k=args.min_k, max_k=args.max_k, order=args.order)