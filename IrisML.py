import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
from random import uniform
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

#Loads the Iris Flower data set.
iris_data_set = load_iris()
iris = pd.DataFrame(data= np.c_[iris_data_set['data'], iris_data_set['target']], 
                    columns= iris_data_set['feature_names'] + ['target'])
x = iris.iloc[:, [1, 2, 3, 4]].values

#Trains the k-neighbor with the Iris Dataset.
x_train, x_test, y_train, y_test = train_test_split(iris_data_set["data"], iris_data_set["target"], random_state = 0)
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train, y_train)

#Cluster creation for iris data set and graph showing the optimal number of clusters using the Elbow Method
wcss = []

for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(iris)
    wcss.append(kmeans.inertia_)

#Creates random numbers to see how they are predicted by the k-neighbors algorithm.
for i in range(10):
    x_new = np.array([[randint(1,5), randint(1,5), randint(1,5), uniform(0.01,1)]])
    prediction = kn.predict(x_new)

    print("Predicted target value: {}\n".format(prediction))
    print("Predicted feature name: {}\n".format(iris_data_set["target_names"][prediction]))
    print("Test score: {:.2f}".format(kn.score(x_test, y_test)))

#Plots graph of optimal number of clusters using the Elbow Method
plt.plot(range(1,11), wcss)
plt.title('Elbow Methor')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS') #WCSS = Within Cluster Sum of Squares
plt.show()

