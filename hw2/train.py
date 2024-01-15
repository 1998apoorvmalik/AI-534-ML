#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector
import numpy as np
from sklearn import svm as SVM

def read_from(textfile):
    for line in open(textfile): 
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    v['<bias>'] = 1
    for word in words:
        v[word] += 1
    return v
    
def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now

def write_test_predictions(model):
    with open('test.txt.predicted', 'w') as f:
        for i, (label, words) in enumerate(read_from('test.txt'), 1):
            f.write("%s\t%s\n" % ("+" if model.dot(make_vector(words)) > 0 else "-", " ".join(words)))
            
def train(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = 1.
    model = svector({'<bias>': 0})
    best_model = model.copy()
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
        dev_err = test(devfile, model)
        if dev_err < best_err:
            best_model = model.copy()
            best_err = dev_err

        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    return best_model

def train_avg_perceptron(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = 1.
    model = svector({'<bias>': 0})
    avg_model, best_model = model.copy(), model.copy()
    c = 1
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                avg_model += c * label * sent
            c += 1
        
        new_model = c * model - avg_model
        dev_err = test(devfile, new_model)
        if dev_err < best_err:
            best_model = new_model.copy()
            best_err = dev_err
    
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

    write_test_predictions(best_model)
    return best_model

def train_svm(trainfile, devfile):

    def vectorize(tokens):
        X = np.zeros(len(all_tokens))
        for token in tokens:
            if token in all_tokens:
                X[all_tokens[token]] += 1
        return X

    train_data = list(read_from(trainfile))
    test_data = list(read_from(devfile))
    all_tokens = {token: i for i, token in enumerate(sorted(list(set([token for label, sentence in train_data for token in sentence]))))}

    # create train data
    X_train, y_train = [], []
    for label, sentence in train_data:
        X_train.append(vectorize(sentence))
        y_train.append(label)
    X_train, y_train = np.array(X_train), np.array(y_train)
    print("[INFO] Train data created\nShape of X_train: {}\nShape of y_train: {}".format(X_train.shape, y_train.shape))

    # create dev data
    X_dev, y_dev = [], []
    for label, sentence in test_data:
        X_dev.append(vectorize(sentence))
        y_dev.append(label)
    X_dev, y_dev = np.array(X_dev), np.array(y_dev)
    print("[INFO] Dev data created\nShape of X_dev: {}\nShape of y_dev: {}".format(X_dev.shape, y_dev.shape))
        
    # train SVM
    print("[INFO] Training SVM...")
    svm = SVM.SVC()
    svm.fit(X_train, y_train)
    print("[INFO] SVM trained")
    print("[INFO] Evaluating SVM...")
    svm.predict(X_dev)
    error = 1 - svm.score(X_dev, y_dev)
    print("\n[INFO] SVM dev error: {:.2f}%".format(error * 100))

if __name__ == "__main__":
    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    option = sys.argv[3]

    if option == '1':
        train(trainfile=sys.argv[1], devfile=sys.argv[2])
    elif option == '2':
        train_avg_perceptron(trainfile=sys.argv[1], devfile=sys.argv[2])
    elif option == '3':
        train_svm(trainfile=sys.argv[1], devfile=sys.argv[2])
    else:
        print("[ERROR] Invalid option, please provide from the following options: 1 (Perceptron), 2 (Average Perceptron), 3 (SVM)")
        sys.exit(1)
