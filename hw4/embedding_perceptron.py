import time
import numpy as np
from gensim.models import KeyedVectors
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--train', default='./train.txt')
argparser.add_argument('--dev', default='./dev.txt')
argparser.add_argument('--test', default='./test.txt')
argparser.add_argument('--epochs', default=10)
argparser.add_argument('--model', type=int, default=1, help='0: perceptron, 1: avg perceptron, 2: svm')


def make_vector(words, word_vectors):
    embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)

def train_basic_perceptron(trainfile, devfile, word_vectors, epochs=10):
    t = time.time()
    best_err = 1.
    # Initialize the model as a vector of zeros
    model = np.zeros(word_vectors.vector_size)
    best_model = model.copy()
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent = make_vector(words, word_vectors)
            if label * np.dot(model, sent) <= 0:
                updates += 1
                model += label * sent
        dev_err = test(devfile, model, word_vectors)
        if dev_err < best_err:
            best_model = model.copy()
            best_err = dev_err

        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    return best_model

def train_avg_perceptron(trainfile, devfile, word_vectors, epochs=10):
    t = time.time()
    best_err = 1.
    model = np.zeros(word_vectors.vector_size)
    avg_model = np.zeros(word_vectors.vector_size)
    c = 1
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent = make_vector(words, word_vectors)
            if label * np.dot(model, sent) <= 0:
                updates += 1
                update = label * sent
                model += update
                avg_model += c * update
            c += 1

        new_model = model - avg_model / c
        dev_err = test(devfile, new_model, word_vectors)
        if dev_err < best_err:
            best_model = new_model.copy()
            best_err = dev_err

        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(best_model), time.time() - t))


    return best_model


def test(devfile, model, word_vectors):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        sentence_embedding = make_vector(words, word_vectors)
        prediction = np.dot(model, sentence_embedding)
        err += label * prediction <= 0
    return err / i  # i is |D| now



def read_from(textfile):
    for line in open(textfile): 
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())


if __name__ == "__main__":
    args = argparser.parse_args()
    wv = KeyedVectors.load('embs_train.kv') 
    
    print("[INFO] Using Sentence Embeddings")
    if args.model == 0:
        print("[INFO] Training basic perceptron...")
        model = train_basic_perceptron(args.train, args.dev, wv, epochs=args.epochs) 
    elif args.model == 1:
        print("[INFO] Training average perceptron...")
        model = train_avg_perceptron(args.train, args.dev, wv, epochs=args.epochs)
        
        # save dev predictions to file
        print("[INFO] Writing dev predictions to file...")
        with open('dev.txt.predicted.embedded.avg_perceptron', 'w') as f:
            for i, (label, words) in enumerate(read_from(args.dev), 1):
                f.write("%s\t%s\n" % ("+" if np.dot(model, make_vector(words, wv)) > 0 else "-", " ".join(words)))