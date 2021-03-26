from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import sys
import gzip #for zipping and unzipping files
import time

n_clusters = int(sys.argv[1])
elements = int(sys.argv[2])
iterations = int(sys.argv[3])


t = []
acc = []
def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    return data / np.float32(256)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        #data2 = np.zeros( (len(data),10), dtype=np.float32 )
        #for i in range(len(data)):
        #    data2[i][ data[i] ] = 1.0
    return data

## Load the training set
TrainX = load_mnist_images('train-images-idx3-ubyte.gz')
TrainY = load_mnist_labels('train-labels-idx1-ubyte.gz')

## Load the testing set
TestX = load_mnist_images('t10k-images-idx3-ubyte.gz')
TestY = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


for i in range(iterations):
    s = time.time()
    X = TrainX
    kmeans = KMeans(n_clusters=n_clusters).fit(X)

    lbls = []
    for p in range(n_clusters):
        lbls.append([i for i, x in enumerate(list(kmeans.labels_)) if x == p])

    chosen = []
    for i in range(n_clusters):
        chosen.append(np.random.choice(lbls[i], elements, replace=False))

    chosen = [item for sublist in chosen for item in sublist]

    X = TrainX[chosen]
    y = TrainY[chosen]
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=15)
    neigh.fit(X, y)
    pred = neigh.predict(TestX)

    t.append(time.time()-s)
    acc.append(accuracy_score(pred, TestY))
print(min(t), mean(t), max(t))
print(min(acc), mean(acc), max(acc))
