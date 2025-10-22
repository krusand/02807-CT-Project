# Related work
This markdown document provides an overview of possible relevant papers for our related work section. The papers might also help us scope our project. 

## Clustering-Based Frequent Pattern Mining Framework for Solving Cold-Start Problem in Recommender Systems
Kannout et. al. 2024. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10401921

- Applies the theory of frequent itemsets
- Includes some nice diagrams providing overview of their methodology 
- Introduces the clustering-based frequent pattern mining framework for recommender systems (Clustering-based FPRS) - a novel RS constituting several recommendation strategies leveraging agglomerative clustering and FP-growth algorithms
- The developed strategies combine the generated frequent itemsets with collaborative- and content-filtering methods to address the cold-start problem
- Difficulties for RS:
	- Density of interactions
	- Dynamic changes in the commercial environment, e.g., when a new product is being launched 
- Main contributions of paper:
	- Introduces *Clustering-based Frequent Pattern mining framework for Recommender Systems* constituting of several recommendation strategies to mitigate the cold-start problem
	- Utilizes the FP-growth algorithm to produce frequent itemsets based on the ratings in the user-item matrix
	- Mitigates the sparsity issue by converting the user-item rating matrix into a lower-dimensional one with clustering techniques 
	- Incorporates contextual information and extend the recommendation list using similarity measures of users and items to address the cold-start problem in the FPRS framework
	- Conducts comprehensive experiments of the proposed approach on several benchmark datasets, including MovieLens 100K, MovieLens 1M, HetRec-MovieLens, and LDOS-CoMoDa, and evaluates Clustering-based FPRS against several state-of-the-art recommender systems
- FP-growth is an algorithm for mining frequent itemsets like the a-priori algorithm
- FP-growth has several advantages over other algorithms
- A-priori suffers from two major shortcomings:
	- Large size of candidate itemsets
	- High costs of disk I/O and computing power due to multiple scans of the database
- All these problems are solved in FP-growth by leveraging the FP-tree data structure to store all data in a concise and compact way
	- Once the FP-tree is constructed, one can directly use a recursive divide-and-conquer approach to efficiently mine the frequent itemsets without any need to scan the database over and over again
- Hard clustering: Each object should belong to one and only one cluster
- Soft/fuzzy clustering: Each object can belong to more than one cluster with probabilities 
- 5 stages for generating recommendations:
	1. Data Input
		- The user-item rating matrix is enriched by selected content-based users' and items' characteristics 
	2. Data Preparation
		- Data quality analysis
		- Filtering
		- Verification of selected features' importance
	3. Data preprocessing
		- The user-item rating matrix is converted into a lower-dimensional one using clustering techniques 
		- The matrix is partitioned based on selected features, resulting in the creation of smaller matrices 
	4. Frequent Pattern Mining
		- Generate frequent itemsets using the FP-growth algorithm
	5. Recommendation Generation 
		- Produce the recommendations with the FPRS framework that consists of two main modules:
			- User cold-start
			- Item cold-start
		- Instead of providing one method that fits all purposes, multiple strategies are followed as part of the framework
		- Each module offers several approaches to selecting the features and producing recommendations
- Recommendation score for user $i$ and item $j$ using fuzzy clustering: $$s(i,j)=a_{i}\cdot b_{j}=\sum_{r=1}^ka_{ir}b_{jr}$$where 
	- $a_i$ is the vector of length $k$ (the number of clusters) containing the probabilities that user $i$ belongs to each cluster
	- $b_j$ is the aggregated score (average, median, or range of min-max values) for item $j$ within each cluster 
- User cold-start module:
	- Strategy 1:
		- Recommend the frequent singletons (the most popular items) to the new user
	- Strategy 2:
		- Split data by user demographics
		- For each user group, use the items in the generated frequent singletons to compose the recommendation set that will be recommended to new users based on their demographics 
	- Strategy 3 + 4:
		- Generate frequent sets of users
		- The items rated by frequent set of users and fulfilled the participation threshold, are utilized to form the recommendation set for a dedicated user group
		- Strategy 3: split by gender of the users
		- Strategy 4: Split the data using users' and items' characteristics, so the recommendation set is created for each user group using the frequent set of users that exist in the group containing the largest number of these frequent sets (the dominant group)
- Item cold-start module:
	- Strategy 5 + 6: 
		- Strategy 5: Split the data only by items' characteristics
		- Strategy 6: Split the data by both items' and users' characteristics
		- Compose the recommendation set by selecting the users who are involved in generating the frequent itemsets
		- Recommend the new item to a set of users based on the item group
		- The users in the recommendation set are selected from all frequent itemsets in Strategy 5
		- In Strategy 6, the users are selected from the user group that contains the largest number of frequent itemsets, called the dominant group, since in this strategy we split the data by users' and items' characteristics 
	- Strategy 7:
		- Form clusters based on the frequent singletons created by users' and items' features 
		- The recommendation set for the new item is produced based on the percentage of users' engagement in creating the frequent itemsets in the closest cluster
- Time complexity of FP-growth to identify all frequent itemsets of size $k$ is $O(n^2)$, where $k$ is the number of elements in the itemset 
	- The complexity decreases when we impose constraints on the size of frequent itemsets, e.g. for $k=1$ (frequent singletons, linear complexity)
- Includes a thresholds sensitivity analysis, showing the F1-score for different thresholds for strategy 5, 6, and 7
	- The highest F1-score is usually obtained for thresholds in the range 0.1-0.2 


## Mining Association Rules for Optimizing Customer Purchase Behavior in E-Commerce Transactions
Siam et. al. 2025. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11172084&casa_token=jJ8Ipu2HFU0AAAAA:Mm9SEyFqf8xRCT2Rc9zJBaZZQvQDA1vZbmKg6hJlwzV7IIuPJAXhiECRmJLFZeF_wlksRcBD&tag=1


## Review of Clustering-Based Recommender Systems 
Beregovskaya Irina and Koroteev Mikhail. 2021. https://arxiv.org/pdf/2109.12839

- A good overview of clustering as a pre-processing step 
- Clustering can be used to solve problems with sparsity of the rating matrix for CF
- Clustering can also be used to effectively solve cold start problems when a new user or a new book, about which little is known, appears in the system
- Idea: Cluster users and apply CF on each cluster (with all items or items potentially clustered as well)


## Automatic Collection Creation and Recommendation 
Singal et. al. 2021. https://arxiv.org/pdf/2105.01004

- Production-style use of clustering/dimensionality reduction to build collections/playlists
- Demonstrates how recommending collections can be turned into item consumption 


## ClusterSeq: Enhancing Sequential Rcommender Systems with Clustering based Meta-Learning 
Maheri et. al. 2023. https://arxiv.org/pdf/2307.13766
- Item/user sequences are clustered and cluster-level signals are used to guide next-item recommendations
- Shows how to map cluster signals back to items using similarity/re-ranking within the cluster
- Useful for sequential settings


## Co-clustering for Federated Recommender System 
He et. al. 2024. https://arxiv.org/pdf/2411.01690v1
- Co-clustering (joint user x item clustering)
- Illustrates practical mappings from co-clusters to item candidates for downstream ranking


## ProtoCF: Prototypical Collaborative Filtering for Few-shot Recommendation
Sankar et. al. 2021. https://aditk2.web.engr.illinois.edu/reports/recsys21.pdf?utm_source=chatgpt.com

- Formalizes using prototypes to represent item clusters and then compose specific item scores from prototypes 


## Personalisation of D'Hondt's algorithm and its use in recommender ecosystems
Balcar et. al. 2024. https://arxiv.org/pdf/2406.07264

- Discusses using D'Hondt to proportionally allocate recommendation slots across clusters and how hybrid personalised/global weights affect which clusters receive slots 


## Overall Reflections
Patterns researchers use to convert cluster/category recommendations into specific item recommendations:
- Recommend the cluster, then show representative items (prototypes/centroids)
	- Idea: pick a small set of representative items (cluster prototypes or items nearest the centroid) and show these as concrete recommendations
	- Pros: cheap, interpretable, fast
	- Cons: might miss user-specific nuance inside cluster
	- Papers: ProtoCF
- Candidate generation by cluster + intra-cluster ranking (local scoring)
	- Idea: clustering is used as candidate generation (reduce search to items in top-K clusters for a user); then within those candidate items you run a local ranking model, e.g., item-to-item CF, learned ranker, or content-based score 
	- How it's derived: 
		1. Compute cluster affinity for user (user $\rightarrow$ cluster score)
		2. Pick top clusters
		3. Fetch items in those clusters
		4. Score items with a ranking model (simple popularity or an ML ranker)
	- Pros: Scalable (cuts matrix size), retains personalization via ranking model
	- Cons: Two-stage pipeline complexity 
	- Papers: 2021 review paper, ClusterSeq, Co-clustering 
- Cluster-level score + item similarity composition (soft decomposition)
	- Idea: the model predicts a user's affinity for clusters and then composes item scores inside a cluster via item-cluster membership weights (soft assignment)
		- Effectively: `score(user,item) ≈ sum_over_clusters [ P(cluster|user) * P(item|cluster) ]`
	- How it's derived:
		- Use soft/fuzzy clustering (e.g., probabilistic mixture models, or learned cluster-to-item distributions) so you can directly compute item scores from cluster probabilities
	- Pros: Principled, maintains uncertainty; can be efficient if cluster count $<<$ item count
	- Cons: requires good soft assignment estimates
	- Papers: Co-clustering, review paper
- Use clusters to re-rank or diversify top-N
	- Idea: produce a candidate item list (from any CF), then use cluster labels to enforce diversity or to re-rank so the final top-N is balanced across clusters (or tailored to pick best item per cluster)
	- How it's derived:
		- Run a ranking objective with a diversity penalty or apply heuristics like "pick top item from top k clusters"
	- Pros: improves diversity and serendipity; practical for UX where users want varied suggestions
	- Cons: may reduce raw accuracy if misapplied 
	- Papers: Automatic collection creation and recommendation
- Build collections (clusters) as a product and surface items after user selection
	- Idea: the system actually recommends collections. If the user taps a collection they are shown the items inside it; click-through is used to personalize which items to surface first. This is common in music/media
	- Exploit interaction signals on collections to bias item ordering
	- Pros: matches product/UX patterns, higher conversion in some domains (e.g. music)
	- Cons: Extra product design overhead
	- Papers: Automatic collection creation and recommendation
