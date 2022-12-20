Importing libraries

%matplotlib inline
import math
from math import pi

import numpy as np
import pandas as pd
import collections

import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score 

from yellowbrick.cluster import KElbowVisualizer
#from yellowbrick.cluster import InterclusterDistance
Loading the dataset

We read the .csv files containing the two sub-datasets.

cs_num = pd.read_csv('dataset/customers_num.csv', index_col=0)
cs_cat =  pd.read_csv('dataset/customers_cat.csv', index_col=0)
Elimination of highly correlated features
We begin by examining the correlations between the attributes of the dataset to be clustered in order to identify the highly correlated couples. Dropping redundant attributes benefits the analysis by reducing the dimensionality of the dataset and rising the influence that more useful feature could have on the whole clustering process.

With such aim in mind, we fix a maximum threshold value in order to identify highly correlated features and subsequently drop them.

corr_threshold = 0.9
print("Att. A\tAtt. B\tCorr(A,B)")
for i in range(0, len(cs_num.columns)):
    for j in range(i+1, len(cs_num.columns)):
        corr = cs_num[cs_num.columns[i]].corr(cs_num[cs_num.columns[j]])
        if  corr > corr_threshold:
            print(cs_num.columns[i] + "\t" + cs_num.columns[j] + "\t" + '{:.4f}'.format(corr))
Att. A	Att. B	Corr(A,B)
I	Stot	0.9362
Iu	NSess	0.9684
Eb	Ew	0.9198
Eb	Em	0.9658
Ew	Em	0.9114
Smax	SWmax	0.9791
Smax	SMmax	0.9229
Savg	SWavg	0.9700
Savg	SMavg	0.9265
SWmax	SMmax	0.9446
SWavg	SMavg	0.9596
As a result of this analysis we decide to eliminate many of the attributes derived from customer spending (SWmax, SWavg, SMmax, SMavg) and some entropy indicators (Em, Ew).

Despite the high correlation between the Stot and I attributes we decide not to eliminate either of them due to their relevance within the whole grouping process. This choice was suggested by the tests mentioned above and will be confirmed in the rest of the notebook.

cs_corr_columns = ['Iavg', 'Ew', 'Em', 'Ir', 'Sref', 'SWmax', 'SWavg', 'SMmax', 'SMavg']
cs_num.drop(cs_corr_columns, axis=1, inplace=True)
cs = cs_num
This is the current arrangement of the dataset.

cs.info()
<class 'pandas.core.frame.DataFrame'>
Float64Index: 3643 entries, 13047.0 to 12713.0
Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   I       3643 non-null   int64  
 1   Iu      3643 non-null   int64  
 2   Imax    3643 non-null   int64  
 3   Ep      3643 non-null   float64
 4   Eb      3643 non-null   float64
 5   Stot    3643 non-null   float64
 6   Smax    3643 non-null   float64
 7   Savg    3643 non-null   float64
 8   NSess   3643 non-null   int64  
dtypes: float64(5), int64(4)
memory usage: 284.6 KB
Normalization
The most common adopted normalizations are: Z-Score and Min-Max.

The Min-Max normalization approach instead allows to have data that are alwayse interpretable, so that we can avoid to apply inverse transformation.
The Z-Score normalization approach exploits the mean and standard deviation of data, and tries to center data with respect to these two statistical properties.
In the coming analysis we choose to use the Min-Max normalization in order to have a leaner approach and avoid an unnecessary inverse transformation.

Min-Max Normalization
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
cs_norm_minmax = minmax_scaler.fit_transform(cs.values)
Extra: Z-Score Normalization
# from sklearn.preprocessing import StandardScaler
# z_scaler = StandardScaler()
# cs_norm_z = z_scaler.fit_transform(cs_no_out.values)
Clustering Analysis by K-means
Identification of the best value of k
The value of k is the main parameter of the algorithm and represents the number of clusters that we want to split the dataset into. There are several techniques to identify the optimal value of said parameter. Ultimately our aim in this context is to run multiple executions of the algorithm, each with a different value for k, and then performing an evaluation analysis.

Elbow Method on the clusters Inertia
In cluster analysis, the elbow method is a heuristic used in determining the optimal number of clusters in a data set.
It consists of plotting the SSE as a function of the number of clusters, and picking the elbow of the curve as the k to use.

We start from an initial value of 2 and we keep computing the clusterization until we reach the value of 30, our choice for the maximum value of k.
Considering ranges of k differing in size may lead to slightly different elbows, so we also plot the curves of the intervals [2,10] and [2,20].

For k = [2, 10], [2, 20] and [2,30]
k_start = [2]
k_max = [10, 20, 30]
Ks = []
model = KMeans()

# print the elbow plots
f, axs = plt.subplots(nrows=1, ncols=len(k_max), figsize=(30,5))

for i in range(len(k_max)):
    vis = KElbowVisualizer(model, k=(k_start[0],k_max[i]), metric='distortion', timings=False, ax=axs[i])
    vis.fit(cs_norm_minmax)  
    axs[i].set_title('Distortion Score Elbow for K-Means Clustering (K = ' + str(k_start[0]) + ', ' + str(k_max[i]) + ')' )
    axs[i].set_ylabel('distortion score')
    axs[i].set_xlabel('k')
    axs[i].legend(['distortion score for k','elbow at k = ' + str(vis.elbow_value_) + ', score = ' + '{:.2f}'.format(vis.elbow_score_)])
    if (vis.elbow_value_ not in Ks) and (vis.elbow_value_ != None):
        Ks.append(vis.elbow_value_)
plt.show()

Extra: Starting by k = 1
In order to experiment with different intervals, we repeat the above analysis starting from a value of k equal to 1, representing the presence of a single large cluster and therefore the absence of the clustering process.

k_start = [1]
k_max = [10, 20, 30]
model = KMeans()

# print the elbow plots
f, axs = plt.subplots(nrows=1, ncols=len(k_max), figsize=(30,5))

for i in range(len(k_max)):
    vis = KElbowVisualizer(model, k=(k_start[0],k_max[i]), metric='distortion', timings=False, ax=axs[i])
    vis.fit(cs_norm_minmax)  
    axs[i].set_title('Distortion Score Elbow for K-Means Clustering (K = ' + str(k_start[0]) + ', ' + str(k_max[i]) + ')' )
    axs[i].set_ylabel('distortion score')
    axs[i].set_xlabel('k')
    axs[i].legend(['distortion score for k','elbow at k = ' + str(vis.elbow_value_) + ', score = ' + '{:.2f}'.format(vis.elbow_score_)])
    if (vis.elbow_value_ not in Ks) and (vis.elbow_value_ != None):
        Ks.append(vis.elbow_value_)
plt.show()

Average Silhouette Method
The silhouette score is a indicator of both separation and cohesion among clusters. Here we look for the clusterization with the highest average silhouette score among those computed by K-means.

k_start = [2]
k_max = [10, 20, 30]
model = KMeans()

# print the elbow plots
f, axs = plt.subplots(nrows=1, ncols=len(k_max), figsize=(30,5))

for i in range(len(k_max)):
    vis = KElbowVisualizer(model, k=(k_start[0],k_max[i]), metric='silhouette', timings=False, ax=axs[i], locate_elbow=False)
    vis.fit(cs_norm_minmax)  
    axs[i].set_title('Silhouette Score Elbow for K-Means Clustering (K = ' + str(k_start[0]) + ', ' + str(k_max[i]) + ')' )
    axs[i].set_ylabel('silhouette score')
    axs[i].set_xlabel('k')
plt.show()

An analysis of the average Silhouette score highlights a global maxima for k equals to 2 and a local maxima for k equals to 4.

k_from_sil = [2,4]
We add these values to the list of candidates for the optimal value of k.

for k in k_from_sil:
    if k not in Ks:
        Ks = np.append(Ks, k_from_sil)
Insights from Hierarchical Clustering
Here we try to identify the optimal number of clusters by analysing the hierarchical clustering performed throught the Ward method.
Said method aims at the local optimization of the SSE fuction and therefore it is often used in conjunction with K-means.

from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering of the Customers")
plt.axhline(y=80, ls='--', c='red')
dend = dendrogram(linkage(cs_norm_minmax, method='ward'), truncate_mode='lastp', p=30, leaf_rotation=60, leaf_font_size = 8, show_contracted=True)

The clustering obtained throught the Ward linkage seems to suggest the presence of three main clusters within the dataset.

k_from_ward = 3
We add this value to the aforementioned list of candidates.

if k_from_ward not in Ks:
    Ks = np.append(Ks, k_from_ward)
We proceed by sorting the list of candidates and perform the clustering for each of these values.

Ks = np.sort(Ks)
Ks
array([2, 3, 4, 5, 7, 8])
Clustering by K-means for each of the candidates
We perform the clustering for the values of k previously picked.
For each k we store the information regarding the centroids, the SSE values associated with the clustering and, most importantly, the labels associating each record to its cluster.

cs_centers = []
cs_labels = []
cs_inertia = []
for k in Ks:
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100) 
    kmeans.fit(cs_norm_minmax) # we perform the clustering for each value of k
    cs_centers.append(minmax_scaler.inverse_transform(kmeans.cluster_centers_)) # we store the coordinates of the centroids
    cs_labels.append(kmeans.labels_) # we store the labels associated with the records
    cs_inertia.append(kmeans.inertia_) # we store the SSE value associated with the clustering
Comparison and evaluation of the different clustering via internal metrics
Each clustering is now evalueated with respect to well known measures of the clusters cohesion and separation. The aim of this process is to identify the value of k resulting in the best defined clusters.

The (internal) evaluation metrics that we are going to compute are:

Sum of Squared Error (SSE)
Davies Bouldin Index
Silhouette Score
Calinski-Harabasz Index, also called Variance Ratio Criterion
int_metrics_K=pd.DataFrame() # we create a dataframe to contain the results of the evaluation

k_cols = []
for k in Ks:
    k_cols.append('K_' + str(k))
    
sse=[]
sep=[]
sil=[]
cal_har=[]

# wrt categorical features
for i in range(len(Ks)):
    sse.append(cs_inertia[i])
    sep.append(davies_bouldin_score(cs_norm_minmax, cs_labels[i]))
    sil.append(silhouette_score(cs_norm_minmax, cs_labels[i]))
    cal_har.append(calinski_harabasz_score(cs_norm_minmax, cs_labels[i]))

int_metrics_K['K'] = k_cols
int_metrics_K['SSE'] = sse
int_metrics_K['Davies_Bouldini'] = sep
int_metrics_K['Silhouette'] = sil
int_metrics_K['Calinski_Harabasz'] = cal_har
int_metrics_K.set_index(['K'], inplace=True)
int_metrics_K
SSE	Davies_Bouldini	Silhouette	Calinski_Harabasz
K				
K_2	557.294186	1.002153	0.435007	3021.802248
K_3	429.136133	1.160057	0.315443	2505.103841
K_4	365.354942	1.192858	0.321506	2172.847202
K_5	314.321795	1.147942	0.272066	2041.370447
K_7	255.594937	1.183484	0.270354	1811.916029
K_8	234.040123	1.127170	0.257634	1743.472511
Regarding the Sum of Squared Errors (SSE):

a decrease in the SSE value proportional to the number of clusters is an expected behavior, therefore opting directly for the clustering with the lowest sum of squared distances may not be a worthwhile decision.
Regarding the Davies Bouldini Index:

a lower Davies-Bouldin index relates to a model with better separation between the clusters and, in this regard, the clustering with k equals to 2 seems to present the best separation among its clusters.
Regarding the Silhouette Score:

a higher value for the Silhouette Coefficient relates to a model with better defined clusters, in this regard, the clustering with k equals to 2 presents the best score.
Regarding the Calinski-Harabasz Index:

similarly to the Silhouette Coefficient, a higher value for the Calinski-Harabasz score relates to a model with better defined clusters, in this regard, the clustering with k equals to 2 seems to present the best defined clusters.
Conclusions
As mentioned previously, a low SSE is not an absolute indicator of good clustering. The purpose of the elbow method is precisely to identify particular values of k for which the decrease in this error is significant even with respect to the normal decrease due to the increase in the number of clusters.

Given the promising results on all of the others evaluation metrics we consider two as the optimal number of clusters for the dataset under analysis.

opt_k = 2
opt_k_index = np.where(Ks == opt_k)[0].item() # we identify the index of the optimal value of k among the other k candidates
opt_k_index
0
