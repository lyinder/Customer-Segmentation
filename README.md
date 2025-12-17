# Customer Segmentation using RFM Analysis

A Python-based customer segmentation solution using RFM (Recency, Frequency, Monetary) analysis combined with machine learning clustering techniques.

## Overview

This project analyzes customer transaction data to segment customers into meaningful groups for targeted marketing and retention strategies. It combines traditional RFM analysis with K-Means clustering and includes demographic predictions.

## Features

- **RFM Analysis**: Calculate Recency, Frequency, and Monetary metrics
- **Customer Segmentation**: Automated clustering into 4 key segments:
  - **High Value**: Best customers with recent purchases and high spend
  - **Active**: Regular customers with good engagement
  - **Infrequent**: Valuable customers with declining activity
  - **Inactive**: Customers at risk of churn
- **Demographic Enhancement**: Gender prediction and age calculation
- **Loyalty Tracking**: Integration of customer loyalty program data
- **Branch Analysis**: Customer home branch identification

## Requirements

```bash
pip install pandas scikit-learn gender-guesser
```

## Dependencies

- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning algorithms (KMeans clustering, StandardScaler)
- `gender-guesser` - Gender prediction from names
- `datetime` - Date and time handling

## Data Structure

The analysis expects a dictionary containing four DataFrames:

1. **states**: Customer master data
   - Columns: `customer_id`, `name`, `dob`, `branch_no`, `createdAt`, `updatedAt`

2. **loyalty**: Loyalty program enrollment data
   - Columns: `account_no`, `createdAt`, `updatedAt`

3. **transactions**: Transaction line items
   - Columns: `loyalty_account`, `date`, `trx_type`, `sale_value`, `qty`, `transaction_id`, `branch`

4. **branches**: Store/branch information
   - Columns: `no`, `name`

## Usage

```python
from RFM_Analysis import execute_analysis
import pandas as pd

# Prepare your data
data_dict = {
    'states': states_df,
    'loyalty': loyalty_df,
    'transactions': transactions_df,
    'branches': branches_df
}

# Run analysis
rfm_results, segment_counts = execute_analysis(
    data_dict=data_dict,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

print(rfm_results.head())
print(segment_counts)
```

## Segmentation Logic

### RFM Scoring (1-4 scale)

- **Recency**: ≤30 days (4), 31-90 (3), 91-180 (2), >180 (1)
- **Frequency**: ≥4 purchases (4), 3 (3), 2 (2), ≤1 (1)
- **Monetary**: ≥$500 (4), $200-499 (3), $50-199 (2), <$50 (1)
- **Loyalty**: ≥200 days (4), 120-199 (3), 60-119 (2), <60 (1)

### Segment Assignment

Customers are assigned to segments based on their RFM scores:
- **High Value**: R≥3, F≥3, M≥3 or Total Score ≥15
- **Active**: (F≥3 or L≥3) and R≥2
- **Infrequent**: (M≥2 or F≥2) and R≤2
- **Inactive**: Low scores across all metrics

## Key Features

### Gender Prediction
Uses title recognition (Mr, Mrs, Ms, Miss) and name-based gender prediction for demographic analysis.

### Age Calculation
Automatically calculates customer age from date of birth, handling edge cases and invalid dates.

### Refund Handling
Properly adjusts monetary values for refunded transactions.

### Missing Branch Resolution
Identifies customer home branches based on transaction frequency when master data is incomplete.

## Output

The function returns:
1. **rfm_data**: DataFrame with customer IDs and assigned segments
2. **segment_counts**: Distribution of customers across segments with timestamps

## Technical Approach

1. **Data Preprocessing**: Clean and standardize customer data, remove test accounts
2. **Feature Engineering**: Calculate RFM metrics from transaction history
3. **Clustering**: Apply K-Means (k=5) with standardized features
4. **Rule-Based Scoring**: Apply business logic for interpretable segments
5. **Enhancement**: Add demographic and branch information

## Skills Demonstrated

- Data wrangling with Pandas
- Machine Learning (K-Means clustering, feature scaling)
- Feature engineering for customer analytics
- Business logic implementation
- Python programming best practices
- Modular and reusable code design

## License

This project is available for portfolio demonstration purposes.

## Author

Lyinder Swale