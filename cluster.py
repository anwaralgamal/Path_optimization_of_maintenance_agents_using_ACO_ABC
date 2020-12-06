import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import xlrd
import csv
import numpy as np


# loading the data from excel file
df = pd.read_excel(r'./data.xlsx')
fig = plt.figure(figsize=(12, 9))
n_clusters=4
total_points=0

# plotting the locations on the curve
locations_lon_x = df['longitude'].tolist()
locations_lat_y = df['latitude'].tolist()
plt.scatter(locations_lon_x, locations_lat_y, s=10, c='blue', marker='o', alpha=1, edgecolors='k', linewidths=1)
plt.grid()
plt.title('Locations of electric scooters', fontsize=20)
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)
plt.show()

# elbow method
dataset = pd.read_excel(r'./data.xlsx')
X = dataset.iloc[:, [0,1]].values
wcss=[]
k_clusters = (1, 10)
for i in range(1,11):
     kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#kmeans
kmeans = KMeans(n_clusters, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0, 1], X[y_kmeans==0, 0], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 1], X[y_kmeans==1, 0], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 1], X[y_kmeans==2, 0], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 1], X[y_kmeans==3, 0], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_kmeans==4, 1], X[y_kmeans==4, 0], s=100, c='magenta', label ='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of scooters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

for m in range(n_clusters):
 e1 = np.vstack((X[y_kmeans==m, 0], X[y_kmeans==m, 1])).T
 seen = set() # check if points are redundant
 e = []
 for item in e1:
     t = tuple(item)
     if t not in seen:
         e.append(list(item))
         seen.add(t)
 print("cluster "+str(m+1)+" : "+str(len(e)+1)+" Points.")
 with open('./dataant/cluster_'+str(m+1)+'.txt','w') as a:
    a.write('32.02 34.84') #start point
    a.write('\n')
    for k in range(len(e)):
            a.write(str(e[k][0]))
            a.write(' ')
            a.write(str(e[k][1]))
            a.write('\n')
 total_points=total_points+len(e)
print("Total Points:"+str(total_points+1))
# save all data of x and y into one txt file             
'''          
with open('./testtt.txt','w') as a:
 for m in range(n_clusters):
  e1 = np.vstack((X[y_kmeans==m, 0], X[y_kmeans==m, 1])).T
  e2=X[y_kmeans==m, 0]
  e3=X[y_kmeans==m, 1]
  seen = set()
  e = []

  for item in e1:
     t = tuple(item)
     if t not in seen:
         e.append(list(item))
         seen.add(t)
  for k in range(len(e)):
            a.write(str(e[k][0]))
            a.write(' ')
            a.write(str(e[k][1]))
            a.write('\n')
'''