
import numpy as np
from src.kNN import KNearestNeighbors
from src.distance import euclidean


# testing the kNN model

raw_data = np.array([[1, 2, 1], [3, 2, 1], [2, 4, 1],
                     [3, 3, 1], [2, 5, 1], [-1, -2, 0],
                     [-3, -2, 0], [-2, -4, 0], [-3, -3, 0],
                     [-2, -5, 0]], dtype=float)

X = raw_data[:, :2]
y = raw_data[:, -1]


model = KNearestNeighbors(k=3, distance_metric=euclidean)
model.train(X, y)

print("Value: {},\tPrediction: {}".format([1, 0], model.predict(np.array([1, 0]))))
print("Value: {},\tPrediction: {}".format([0, 1], model.predict(np.array([0, 1]))))
print("Value: {},\tPrediction: {}".format([0, 0], model.predict(np.array([0, 0]))))
print("Value: {},\tPrediction: {}".format([-1, 0], model.predict(np.array([-1, 0]))))
print("Value: {},\tPrediction: {}".format([0, -1], model.predict(np.array([0, -1]))))
