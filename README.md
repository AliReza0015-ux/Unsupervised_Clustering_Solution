# Unsupervised_Clustering_Solution
Mall Customer Segmentation Model
This project applies K-Means Clustering to segment mall customers based on their behaviors. Using the mall_customers.csv dataset, the goal is to find optimal customer groupings to support targeted marketing strategies.
________________________________________
Project Overview
•	Technique: K-Means Clustering
•	Dataset: Mall customers dataset (includes features like age, annual income, and spending score)
•	Goal: Group customers into meaningful clusters for marketing insights
•	Evaluation Metric: Silhouette Score (used to assess clustering performance)
________________________________________
 Key Results
•	Multiple values of k were tested to evaluate cluster quality using silhouette scores.
•	The optimal number of clusters was found to be 6, based on the highest silhouette score.
•	KMeans++ initialization was used to smartly position centroids by considering data distribution patterns.
Clusters (k)	Silhouette Score
3	0.312
4	0.405
5	0.397
6	0.452
7	0.437
8	0.374
________________________________________
Visualizations
•	Silhouette Scores plotted against different values of k to choose the optimal number of clusters.
•	Cluster visualization to interpret customer segmentation visually.
________________________________________
Technologies Used
•	Python 3
•	Pandas, NumPy
•	Matplotlib, Seaborn
•	Scikit-learn (for KMeans, silhouette_score)
________________________________________
 How to Run
1.	Clone this repository or download the notebook.
2.	Make sure you have Python and the required libraries installed.
3.	Load the dataset: mall_customers.csv
4.	Run the notebook classification_model.ipynb or mall_segmentation.ipynb
5.	Experiment with different values of k or use init='k-means++' as shown:
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)
kmeans.fit(X)
________________________________________
Conclusion
The use of KMeans++ improved centroid initialization, and silhouette analysis helped in selecting the best k. With k=6, the clustering results were most consistent and meaningful for customer segmentation.
 Folder Structure
📁 mall-customer-segmentation/
│
├── mall_customers.csv
├── mall_segmentation.ipynb
├── README.md
________________________________________
Author
Ali Reza Mohseni
https://github.com/AliReza0015-ux
##  Live Demo
[Click to view the deployed app](https://unsupervisedclusteringsolution-f2i9mriduanzmgcnbhjzdd.streamlit.app/)


