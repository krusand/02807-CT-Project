# Market Basket Analysis Plan

## Methods to use
**Apriori** and **PCY** algorithms  
(because the project description specifically mentions “frequent items” as one of the course topics and lists *Apriori and PCY* as examples).

---

## Metrics to evaluate the rules

| Metric | What it means | Why use it |
|--------|----------------|--------------|
| **Support** | How often a combination of products appears in all transactions. | It shows how popular or frequent an itemset is in the dataset. |
| **Confidence** | How often product B is bought when product A is bought. | It measures how strong and reliable a rule is (e.g., if people buy milk, how likely they buy beer too). |
| **Lift** | How much more likely products A and B are bought together compared to random chance. | It helps detect real associations between products (Lift > 1 means a positive relation). |
| **Conviction** | Measures how strongly the absence of A implies the absence of B. | It shows the reliability and direction of the rule beyond chance. |

**Jaccard similarity** could add value,  
because it measures the overall similarity between products based on how often they appear together versus separately, giving a broader view of co-occurrence patterns.

---

## Data preparation steps

From the Instacart dataset, I will use:

- `orders.csv` (contains `user_id`, `order_id`)
- `order_products__prior.csv` (contains `order_id`, `product_id`)
- `products.csv` (contains `product_id`, `product_name`)

The goal is to build a table where each row shows **which user bought which product**, like:

user_id | order_id | product_name


Then group the data:

1. If I want to see **which products are bought together**, I will group by `order_id`.  
2. If I want to see **what each user usually buys**, I will group by `user_id`.

---

## Generating and evaluating the rules

From these grouped baskets, the algorithm will find **frequent itemsets**,  
and then create rules like:

> If a person buys *milk*, they are likely to buy *beer*.

Finally, evaluate each rule using the metrics mentioned above:

- **Support** → how often the pair appears  
- **Confidence** → how reliable the rule is  
- **Lift** → how strongly the products are related  


# Advanced Extensions for the Market Basket Analysis Project

## Clustering of Products or Users

This extension applies clustering algorithms to the co-occurrence matrix, which shows how often two products are bought together.  
By clustering either products or users, it becomes possible to identify groups of similar products or groups of customers with comparable purchasing habits.  
KMeans, DBSCAN, or Hierarchical Clustering can be used.  

## Recommender System Extension

This extension converts the association rules into direct product recommendations.  
For instance, if the rule “milk → beer” has a high confidence value (e.g., above 0.7), then cereal can be suggested to users who have purchased milk.  

## Dimensionality Reduction (PCA or SVD)

Dimensionality reduction techniques such as PCA or SVD can be applied to the basket matrix.  
They reduce the large number of product dimensions into a smaller space that captures the most relevant information.  

## Locality Sensitive Hashing (LSH) `[Topic that have already been covered in the course]`

Locality Sensitive Hashing can be used to find similar baskets or users efficiently without comparing every pair directly.  
It maps similar transactions into the same hash buckets, enabling fast similarity searches.  

## Visualization of Frequent Pairs

Visualization helps interpret the results of Market Basket Analysis more intuitively.  
A network graph can be created to show which products are often bought together.  
Additionally, heatmaps or bar plots can be used to represent metrics such as Lift or Support, making it easier to observe strong associations at a glance.

## Comparison Across Product Categories

The analysis can be expanded by comparing patterns between different product categories, such as “dairy” versus “bakery.”  
This involves grouping the results by aisle or department and calculating average metrics like mean Lift or mean Support for each category.  
It provides insights into which departments have stronger internal associations or which cross-category relationships are most significant.
