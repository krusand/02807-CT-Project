# Project Overview




# Market Basket Analysis
Orders (Basket) - Products (Items)
- Make Rules

# Clustering
- Cluster on User ID using different orders
- Cluster orders using embeddings of products using BERT or Word2Vec

# RecSys
- Make collaborative filtering




# Possible projects
- Graph cluster of warehouse aisle (optimization problem)
- Recommendation system for discounts
- Recommendation system using word2vec for recommending multiple similar items
- Clustering users for (personal) ads, used by Instacart for bulk buying products. Looking up prices.
- Clustering orders for recsys
- Clustering using prices
- RQ: What product should we put on discount for the highest probability for a user to buy, depending on week
- Recommendation system comparison using aisle id compared with using word2vec/BERT and cosine similarity with threshold to categorize products. (Baseline for both: Recommend the most popular item in aisle/embedding group)
- 

# Chosen project
- Recommend product using one of the following methods:
    - Market Basket Analysis using Association rules. Choose the products with high lift
    - Recommendation system using CF with item's aisle id's instead of item id
    - Recommendation system using CF with item's cluster id (based on our clusters)
    - Get ratio of rec/rand using either:
        - Bought before ratio of new vs already bought
        - Clustering users based on frequency, baskets, other features, and use the ratio of bought before

- Evaluation metrics/goal of the recommendations:
    - Precision or NDCG 
    - Reasonable to compute based on the data available
    - Motivated by the goal of providing the customers a fast and efficient e-shopping experience in which they want to find the items they need as fast as possible
    - Alternative goals:
        - Diversity
        - Encouraging the customer's to browse the website and spend more time looking for items of interest
        - Exposing the customer's to new items they might not be aware of 


## Details to figure out/define
- How to construct scores matrix for the CF algorithm?
- How to find the proportion between random recommendations and recommendations for products that the customer has bought before?
- How to define a random recommendation?
- How to convert an asile/cluster recommendation to a product recommendation? 
    - When recommending a known item to the user, it could be the user's most bought item from that aisle



## Rating Metric
Description of the rating we have come up with: $$r_{ui}=w_1\cdot\text{TF.IDF}_{ui}+w_2\cdot\text{recency}_{ui}+w_3\cdot\text{basketfreq}_{ui}$$
Each submetric is defined like this:

### TF.IDF
- Should capture the proportion of item $i$ among all the products that user $u$ has bought as well as the product's overall popularity. 
- The terms are the individual products.
- The documents are the combined orders of each user.
    - Example: 
        - Basket 1: [apple, milk, bread]
        - Basket 2: [apple, bread]
        - Basket 3: [milk]
        - Document (combined orders): [apple, milk, bread, apple, bread, milk]
- Formula:
$$\text{TF.IDF}_{ui}=\frac{n_{ui}}{N_u}\cdot\log\left(\frac{|D_i|}{|D|}\right)$$
- $n_{ui}$: The number of times item $i$ occurs in the document of user $u$
- $N_u$: The total number of items in the document of user $u$
- $|D_i|$: The number of documents containing item $i$ across all users
- $|D|$: The total number of documents (equivalent to the toal number of users)

### Recency
- Takes into account how many days ago user $u$ bought item $i$.
- Computes a weight according to the number of days.
- Lower weights are assigned to items that were bought a long time ago.
- Higher weights are assigned to items were bought recently. 
- $\lambda$ controls the degree of weight decay as the number of days since the product was bought increases.
- Formula: $$\text{recency}_{ui}=\exp(-\lambda\cdot\text{days\_since\_last\_order}_{ui})$$
- $\text{days\_since\_last\_order}_{ui}$: The number of days since user $u$ placed an order containing item $i$
- $\lambda$ is a hyperparameter that needs to be tuned. 

### Basket Frequency
- Accounts for vanishing signals (compared to the TF definition above) when multiple unique items are bought as a part of the same order.
- Formula: $$\text{basketfreq}_{ui}=\frac{|B_{ui}|}{|B_u|}$$
- $|B_{ui}|$: Number of baskets for user $u$ containing item $i$.
- $|B_u|$: Total number of baskets (orders) for user $u$.












