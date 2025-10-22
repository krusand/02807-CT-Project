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


















