import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from knn_classifier import KNNClassifier


def evaluate_sklearn_knn(X_train, y_train, X_test, y_test):
    k_values = np.arange(1, 100, 2)
    k_values = list(k_values) + [len(X_train)]
    run_times, train_positive_rates, dev_positive_rates = [], [], []

    train_errors, dev_errors = [], []
    print("[k]\t[Train Error]\t[Dev Error]\t[Train Positive Rate]\t[Dev Positive Rate]\t[Run Time (ms)]")
    for k in k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)
        start_time = time.time()
        train_error = 1 - knn_classifier.score(X_train, y_train)
        dev_error = 1 - knn_classifier.score(X_test, y_test)
        train_positive_rates.append(np.mean(knn_classifier.predict(X_train) == np.array([0, 1])))
        dev_positive_rates.append(np.mean(knn_classifier.predict(X_test) == np.array([0, 1])))
        run_times.append((time.time() - start_time)* 1000)
        train_errors.append(train_error)
        dev_errors.append(dev_error)
        print("{}\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t\t{:.3f}\t\t\t{:.0f}".format(k, train_error, dev_error, train_positive_rates[-1], dev_positive_rates[-1], run_times[-1]))

    # print min train error and corresponding k
    idx = np.argmin(train_errors[:-1])
    print("min train error = {:.3f} at k = {}".format(train_errors[idx], k_values[idx]))
    # print min dev error and corresponding k
    idx = np.argmin(dev_errors[:-1])
    print("min dev error = {:.3f} at k = {}\n".format(dev_errors[idx], k_values[idx]))


    # print max train error and corresponding k
    idx = np.argmax(train_errors[:-1])
    print("max train error = {:.3f} at k = {}".format(train_errors[idx], k_values[idx]))
    # print max dev error and corresponding k
    idx = np.argmax(dev_errors[:-1])
    print("max dev error = {:.3f} at k = {}\n".format(dev_errors[idx], k_values[idx]))

 
    # print avg train error
    print("avg train error = {:.3f}".format(np.mean(train_errors[:-1])))
    # print avg dev error
    print("avg dev error = {:.3f}\n".format(np.mean(dev_errors[:-1])))

    # print k = inf stats
    print("k=inf\t\ttrain_error={:.3f}\tdev_error={:.3f}\t\ttrain_time={:.3f}ms".format(train_errors[-1], dev_errors[-1], run_times[-1]))


    # plot the train and dev error
    plt.plot(k_values[:-1], train_errors[:-1], label="train error")
    plt.plot(k_values[:-1], dev_errors[:-1], label="dev error")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.xticks(k_values[:-1:4])
    plt.yticks(np.arange(0, 0.25, 0.03))
    plt.grid()
    plt.legend()
    plt.show()

    # plot the train time
    plt.plot(k_values[:-1], run_times[:-1])
    plt.xlabel("k")
    plt.ylabel("train time (ms)")
    plt.xticks(k_values[:-1:4])
    plt.grid()
    plt.show()

    # plot the train and dev positive rates
    plt.plot(k_values[:-1], train_positive_rates[:-1], label="train positive rate")
    plt.plot(k_values[:-1], dev_positive_rates[:-1], label="dev positive rate")
    plt.xlabel("k")
    plt.ylabel("positive rate")
    plt.xticks(k_values[:-1:4])
    plt.grid()
    plt.legend()
    plt.show()


def evaluate_custom_knn(X_train, y_train, X_test, y_test, order=2, min_k=1, max_k=100):
    k_values = np.arange(min_k, max_k, 2)
    k_values = list(k_values) + [len(X_train) - 1]
    run_times, train_positive_rates, dev_positive_rates = [], [], []

    train_errors, dev_errors = [], []
    print("[k]\t[Train Error]\t[Dev Error]\t[Train Positive Rate]\t[Dev Positive Rate]\t[Run Time (ms)]")
    for k in k_values:
        knn_classifier = KNNClassifier(k)
        knn_classifier.fit(X_train, y_train)
        start_time = time.time()

        train_predictions = knn_classifier.predict(X_train, order=order)
        dev_predictions = knn_classifier.predict(X_test, order=order)

        train_error = 1 - np.mean(train_predictions == y_train)
        dev_error = 1 - np.mean(dev_predictions == y_test)

        train_positive_rates.append(np.mean(train_predictions == np.array([0, 1])))
        dev_positive_rates.append(np.mean(dev_predictions == np.array([0, 1])))
        
        run_times.append((time.time() - start_time) * 1000)
        train_errors.append(train_error)
        dev_errors.append(dev_error)
        print("{}\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t\t{:.3f}\t\t\t{:.0f}".format(k, train_error, dev_error, train_positive_rates[-1], dev_positive_rates[-1], run_times[-1]))

    #   print min train error and corresponding k
    idx = np.argmin(train_errors[:-1])
    print("min train error = {:.3f} at k = {}".format(train_errors[idx], k_values[idx]))
    # print min dev error and corresponding k
    idx = np.argmin(dev_errors[:-1])
    print("min dev error = {:.3f} at k = {}\n".format(dev_errors[idx], k_values[idx]))


    # print max train error and corresponding k
    idx = np.argmax(train_errors[:-1])
    print("max train error = {:.3f} at k = {}".format(train_errors[idx], k_values[idx]))
    # print max dev error and corresponding k
    idx = np.argmax(dev_errors[:-1])
    print("max dev error = {:.3f} at k = {}\n".format(dev_errors[idx], k_values[idx]))

 
    # print avg train error
    print("avg train error = {:.3f}".format(np.mean(train_errors[:-1])))
    # print avg dev error
    print("avg dev error = {:.3f}\n".format(np.mean(dev_errors[:-1])))

    # print k = inf stats
    print("k=inf\t\ttrain_error={:.3f}\tdev_error={:.3f}\t\ttrain_time={:.3f}ms".format(train_errors[-1], dev_errors[-1], run_times[-1]))


    # plot the train and dev error
    plt.plot(k_values[:-1], train_errors[:-1], label="train error")
    plt.plot(k_values[:-1], dev_errors[:-1], label="dev error")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.xticks(k_values[:-1:4])
    plt.yticks(np.arange(0, 0.25, 0.03))
    plt.grid()
    plt.legend()
    plt.show()

    # plot the train time
    plt.plot(k_values[:-1], run_times[:-1])
    plt.xlabel("k")
    plt.ylabel("train time (ms)")
    plt.xticks(k_values[:-1:4])
    plt.grid()
    plt.show()

    # plot the train and dev positive rates
    plt.plot(k_values[:-1], train_positive_rates[:-1], label="train positive rate")
    plt.plot(k_values[:-1], dev_positive_rates[:-1], label="dev positive rate")
    plt.xlabel("k")
    plt.ylabel("positive rate")
    plt.xticks(k_values[:-1:4])
    plt.grid()
    plt.legend()
    plt.show()
