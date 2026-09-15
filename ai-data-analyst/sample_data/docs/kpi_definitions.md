# KPI Definitions

This document defines the key business metrics referenced when
analyzing the sales dataset.

## Revenue Growth Rate
The percentage change in total revenue between two comparable periods
(e.g., month-over-month or quarter-over-quarter). Calculated as
(current period revenue - prior period revenue) / prior period revenue.

## Profit Margin
Net profit divided by revenue, expressed as a percentage. This is the
primary measure of how efficiently revenue is converted into profit.
A healthy target profit margin for this business is 15-25%.

## Customer Churn Rate
The percentage of customers who churned (see `churned` column) out of
all customers in a given period. Churn is considered a lagging
indicator of customer satisfaction and discount strategy problems.

## At-Risk Customer
Any customer with a `customer_satisfaction` score below 2.5. At-risk
customers are prioritized for retention outreach by the customer
success team.

## Discount Efficiency
A qualitative assessment of whether higher discounts are actually
driving proportionally higher unit sales, or simply eroding margin
without meaningfully increasing volume. Historically, discounts above
15% have shown diminishing returns for this business.

## Outlier Order
Any order whose revenue is more than roughly 3x the typical order size
for its region and segment. Outlier orders are often legitimate bulk
purchases but should be reviewed separately from standard retail
transactions when computing average order value, since they can skew
averages significantly.
