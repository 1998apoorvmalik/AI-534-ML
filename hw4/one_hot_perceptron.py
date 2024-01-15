#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector
import numpy as np
from sklearn import svm as SVM
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--train', default='./train.txt')
argparser.add_argument('--dev', default='./dev.txt')
argparser.add_argument('--test', default='./test.txt')
argparser.add_argument('--epochs', default=10)
argparser.add_argument('--model', type=int, default=1, help='0: perceptron, 1: avg perceptron')

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
            
def train_basic_perceptron(trainfile, devfile, epochs=10):
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

    return best_model


if __name__ == "__main__":
    args = argparser.parse_args()

    print("[INFO] Using One-Hot Vectors")
    if args.model == 0:
        print("[INFO] Training basic perceptron...")
        train_basic_perceptron(args.train, args.dev, args.epochs)
    elif args.model == 1:
        print("[INFO] Training average perceptron...")
        best_model = train_avg_perceptron(args.train, args.dev, args.epochs)

        # save dev predictions to file
        # print("[INFO] Writing dev predictions to file...")
        # with open('dev.txt.predicted.one_hot.avg_perceptron', 'w') as f:
        #     for i, (label, words) in enumerate(read_from(args.dev), 1):
        #         f.write("%s\t%s\n" % ("+" if best_model.dot(make_vector(words)) > 0 else "-", " ".join(words)))
        

    else:
        print("[ERROR] Invalid option, please provide from the following options: 0 (Perceptron), 1 (Average Perceptron), 2 (SVM)")
        sys.exit(1)
