#!/usr/bin/env python3

import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.metrics import accuracy_score
import time
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--train', default='./train.txt')
argparser.add_argument('--dev', default='./dev.txt')
argparser.add_argument('--test', default='./test.txt')
argparser.add_argument('--run-test', action='store_true', default=False)

def read_from(textfile):
    for line in open(textfile): 
        label, words = line.strip().split("\t")
        yield (1 if label == "+" else -1, words.split())

def make_vector(words, word_vectors):
    embeddings = [word_vectors[word] for word in words if word in word_vectors]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word_vectors.vector_size)

def prepare_data(file_path, word_vectors):
    X, y = [], []
    for label, words in read_from(file_path):
        X.append(make_vector(words, word_vectors))
        y.append(label)
    return np.array(X), np.array(y)

def generate_test_predictions(testfile, model, word_vectors, output_file):
    with open(testfile, 'r') as file, open(output_file, 'w') as out_file:
        for line in file:
            words = line.strip().split()
            words.pop(0)    # remove the ? label
            sentence_embedding = make_vector(words, word_vectors)
            prediction = model.predict([sentence_embedding])[0]
            label = '+' if prediction == 1 else '-'
            out_file.write(f"{label}\t{' '.join(words)}\n")

if __name__ == "__main__":
    args = argparser.parse_args()
    wv = KeyedVectors.load('embs_train.kv', mmap='r')

    # Prepare training and development data
    X_train, y_train = prepare_data(args.train, wv)
    X_dev, y_dev = prepare_data(args.dev, wv)
    
    # Train SVM model
    start_time = time.time()
    print("[INFO] Training SVM...")
    model = svm.SVC(C=1, gamma=1.32, kernel='rbf')
    model.fit(X_train, y_train)
    print("[INFO] SVM trained")
    print("[INFO] Training time: {:.2f} seconds".format(time.time() - start_time))

    # Evaluate the model
    start_time = time.time()
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_dev)
    error = 1 - accuracy_score(y_dev, y_pred)
    print("[INFO] Dev error: {:.4f}%".format(error * 100))
    print("[INFO] Evaluation time: {:.2f} seconds".format(time.time() - start_time))

    if args.run_test:
        file_name = 'test.txt.predicted'
        print("[INFO] Generating predictions for the test dataset...")
        generate_test_predictions(args.test, model, wv, file_name)
        print("[INFO] Test predictions saved to {}".format(file_name))