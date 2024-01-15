import numpy as np
from gensim.models import KeyedVectors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--train', default='./train.txt')
argparser.add_argument('--dev', default='./dev.txt')
argparser.add_argument('--test', default='./test.txt')
argparser.add_argument('--mode', type=int, default=1, help='0: one-hot vector, 1: sentence embedding')

def compute_sentence_embedding(sentence, word_vectors):
    embeddings = []
    for word in sentence.split():
        if word in word_vectors:
            embeddings.append(word_vectors[word])
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word_vectors.vector_size)

def load_sentences_and_labels(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label, sentence = line.strip().split('\t')
            sentences.append(sentence)
            labels.append(1 if label == '+' else 0)
    return sentences, labels


if __name__ == '__main__':
    args = argparser.parse_args()
    
    # Load pre-trained word vectors
    wv = KeyedVectors.load('embs_train.kv')

    # Load training and development data
    train_sentences, train_labels = load_sentences_and_labels('train.txt')
    dev_sentences, dev_labels = load_sentences_and_labels('dev.txt')

    if args.mode == 0:
        # Convert sentences to one-hot vectors
        vectorizer = CountVectorizer(binary=True)  # binary=True for one-hot encoding
        X_train = vectorizer.fit_transform(train_sentences)
        X_dev = vectorizer.transform(dev_sentences)

        # Convert to dense arrays if necessary
        X_train = X_train.toarray()
        X_dev = X_dev.toarray()

    else:
        # Compute sentence embeddings
        train_embeddings = np.array([compute_sentence_embedding(s, wv) for s in train_sentences])
        dev_embeddings = np.array([compute_sentence_embedding(s, wv) for s in dev_sentences])

    # k-NN classifier for different values of k
    error_rates = {}

    if args.mode == 0:
        for k in range(1, 100, 2):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, train_labels)
            predictions = knn.predict(X_dev)
            error_rate = 1 - accuracy_score(dev_labels, predictions)
            error_rates[k] = error_rate
            print('[LOG] k =', k, '\t dev error rate =', round(error_rate, 4))

    else:
        for k in range(1, 100, 2):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_embeddings, train_labels)
            predictions = knn.predict(dev_embeddings)
            error_rate = 1 - accuracy_score(dev_labels, predictions)
            error_rates[k] = error_rate
            print('[LOG] k =', k, '\t dev error rate =', round(error_rate, 4))

    # Find the best k
    best_k = min(error_rates, key=error_rates.get)
    print('Best k:', best_k)

    # print the best error rate
    print('Best error rate:', error_rates[best_k])