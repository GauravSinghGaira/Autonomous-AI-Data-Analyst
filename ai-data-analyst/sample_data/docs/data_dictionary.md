# Data Dictionary — sales_data.csv

This document describes each column in the sample sales dataset used by
the Autonomous AI Data Analyst demo.

## Columns

**order_date**
The calendar date the order was placed. Daily granularity, ranges over
2024.

**region**
The sales region the order was fulfilled from. One of: North, South,
East, West. Regions are used for territory-level performance reporting.

**segment**
The customer segment that placed the order. One of: Consumer, Corporate,
Home Office. Segment is used to tailor marketing and discount strategy.

**revenue**
Total order revenue in USD, before returns or refunds. This is the
top-line figure used in all revenue reporting.

**profit**
Net profit in USD after cost of goods sold. Profit can be negative for
heavily discounted orders. Profit margin is calculated as profit / revenue.

**units_sold**
Number of individual units included in the order.

**discount_pct**
The percentage discount applied to the order, one of 0, 5, 10, 15, or 20.
Higher discounts are associated with lower customer satisfaction scores
in this dataset, which is a known business concern.

**customer_satisfaction**
A 1-5 customer satisfaction score collected via post-purchase survey.
Some values are missing where the customer did not respond to the
survey. A score below 2.5 is considered "at risk."

**churned**
Binary flag (0/1) indicating whether the customer associated with this
order churned (stopped purchasing) within 90 days. Derived internally
from the satisfaction score and repurchase history.

## Data Quality Notes

- A small number of rows (roughly 1-2%) contain unusually high revenue
  values relative to the rest of the dataset; these should be reviewed
  before being included in forecasting models, as they may represent
  bulk/wholesale orders rather than typical transactions.
- `customer_satisfaction` has missing values for surveys that were not
  completed. Analyses using this column should account for missing data
  rather than assuming a value.
- A handful of duplicate rows exist in the raw export due to a known
  upstream logging issue; deduplication is recommended before financial
  reporting.
