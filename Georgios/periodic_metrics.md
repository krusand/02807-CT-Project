# Periodicity Metrics – Explanation and Rationale

## 1. Purpose of the Metrics
The goal of this analysis is to measure how regularly each user places orders in the Instacart dataset.  
To quantify this **purchase periodicity**, we use statistical measures that describe both the **average frequency** and the **stability** of user behavior across multiple orders.

The following metrics were selected because they are simple, interpretable, and effective in identifying behavioral patterns in time-series–like purchase data.

---

## 2. Mean of `days_since_prior_order`
**Definition:**  
The mean represents the **average number of days** between consecutive orders for a given user.

**Formula:**  
$$
\text{mean} \;=\; \frac{1}{N}\sum_{i=1}^{N} x_i
$$
where $x_i$ is the days-between-orders interval and \(N\) is the number of intervals.

**Interpretation:**  
- A **low mean** indicates a frequent shopper (e.g., orders every 3–5 days).  
- A **high mean** indicates an occasional shopper (e.g., orders every 20–30 days).

**Why it is used:**  
It captures the **overall purchase frequency**, providing a baseline indicator of how often a user returns to buy products.

---

## 3. Standard Deviation of `days_since_prior_order`
**Definition:**  
The standard deviation (std) measures the **variability** of purchase intervals for each user — how much their order timing fluctuates around the average.

**Formula:**  
$$
\text{std} \;=\; \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\left(x_i - \bar{x}\right)^2}
$$
where $ \bar{x} $ is the mean defined above.

**Interpretation:**  
- A **low std** means the user orders at nearly fixed intervals (high regularity).  
- A **high std** means the user orders irregularly (large gaps or bursts of activity).

**Why it is used:**  
While the mean shows how often someone shops, the std shows **how consistent** that rhythm is.

---

## 4. Coefficient of Variation (CV)
**Definition:**  
The coefficient of variation is the ratio of standard deviation to the mean:

The CV scales dispersion by the mean, making users comparable across frequencies:
$$
\mathrm{CV} \;=\; \frac{\text{std}}{\text{mean}}
$$

It expresses variability as a **relative measure**, independent of the absolute order frequency.

**Interpretation:**  
- **Low CV (< 0.5):** stable, predictable behavior.  
- **Moderate CV (0.5–1.0):** some variation, moderately stable.  
- **High CV (> 1.0):** irregular, unpredictable purchase behavior.

**Why it is used:**  
- CV allows comparison between users with different average order frequencies.  
- It normalizes the dispersion, making it possible to identify *patterns of consistency* rather than absolute timing.  
- It is widely used in behavior
