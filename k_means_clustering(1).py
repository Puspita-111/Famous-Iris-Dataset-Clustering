#!/usr/bin/env python
# coding: utf-8

# # Puspita Saha

# # K-Means Clustering

# ## Importing the libraries

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_visual_analysis import VisualAnalysis


# In[8]:


dataset= sns.load_dataset('iris')


# In[10]:


VisualAnalysis(dataset)


# ## Importing the dataset

# In[14]:



X = dataset.iloc[:, [3, 4]].values


# ## Using the elbow method to find the optimal number of clusters

# In[15]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ## Training the K-Means model on the dataset

# In[18]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# ## Visualising the clusters

# In[22]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of species')
plt.legend()
plt.show()


# In[ ]:




