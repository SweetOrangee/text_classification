import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pickle

class MaxEnt(object):
    def __init__(self, M='auto', max_iter=100):
        self.max_iter = max_iter
        self.M = M
    
    def fit(self, X, y, test_X, test_y):
        stops = set(stopwords.words("english"))
        self.vectorizer = CountVectorizer(stop_words=stops)
        X = self.vectorizer.fit_transform(X)
        test_X = self.vectorizer.transform(test_X)
        assert X.shape[0] == len(y)

        self.num_texts, self.num_features = X.shape
        self.num_labels = len(list(set(y)))

        self.Pxy = np.zeros((self.num_labels, self.num_features))
        self.Px = np.zeros(self.num_features)
        self.weights = np.zeros((self.num_labels, self.num_features))

        max_features = 0
        for i in tqdm(range(self.num_texts)):
            row = X.getrow(i)
            label = y[i]
            for index, count in zip(row.indices, row.data):
                self.Pxy[label, index] += 1
                self.Px[index] += 1
            if len(row.indices) > max_features:
                max_features = len(row.indices)
        if self.M == 'auto':
            self.M = max_features
        print(self.M)
        self.Pxy /= self.num_texts
        self.Px /= self.num_texts

        max_acc = 0
        tol_count = 0
        for i in range(self.max_iter):
            print('start iter {}'.format(i))
            ep = self.Ep(X)
            self.last_weights = self.weights.copy()
            
            delta = np.zeros_like(self.Pxy)
            mask = np.logical_and(ep > 1e-10, self.Pxy > 1e-10)
            delta[mask] = 1.0 / self.M * np.log(self.Pxy[mask] / ep[mask])
            self.weights += delta
            diff = np.max(np.abs(delta))
            if i % 5 == 0 and i != 0:
                pred = self.predict(test_X)
                acc = accuracy_score(test_y, pred)
                print('iter {}, acc {}'.format(i, acc))
                if acc > max_acc:
                    max_acc = acc
                    tol_count = 0
                    self.save_model('../checkpoint/maxent_M{}_iter{}_acc{:.2}.pkl'.format(self.M, i, acc))
                else:
                    tol_count += 1
                    if tol_count >= 2:
                        break
            if diff < 1e-5:
                break
    
    def predict_texts(self, texts):
        features = self.vectorizer.transform(texts)
        return self.predict(features)
    
    def predict_texts_prob(self, texts):
        features = self.vectorizer.transform(texts)
        return self.predict_prob(features)

    def Ep(self, X):
        ep = np.zeros((self.num_labels, self.num_features))
        for i in tqdm(range(self.num_texts)):
            features = X.getrow(i)
            prob = self.calculate_prob(features)
            ep[:, features.indices] += np.outer(prob, self.Px[features.indices])
        return ep
    
    def _diff(self, last_weights, weights): 
        return np.max(np.abs(last_weights - weights))
            
    def calculate_prob(self, features):
        related_weights = self.weights[:, features.indices]
        logits = np.sum(related_weights, axis=1)
        prob = logits - np.max(logits)
        prob = np.exp(prob)
        prob = prob / np.sum(prob)
        return prob

    def predict_prob(self, X):
        prob = []
        num_samples = X.shape[0]
        for i in range(num_samples):
            features = X.getrow(i)
            prob.append(self.calculate_prob(features))
        prob = np.array(prob)
        return prob

    def predict(self, X):
        prob = self.predict_prob(X)
        pred = np.argmax(prob, axis=1)
        return pred
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
