#!/usr/bin/env python3

import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import gender_guesser.detector as gender
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def execute_analysis(data_dict, start_date=None, end_date=None, **kwargs):
    """
    Execute RFM (Recency, Frequency, Monetary) analysis on customer transaction data.
    
    Args:
        data_dict: Dictionary containing DataFrames with keys:
                   'states', 'loyalty', 'transactions', 'branches'
        start_date: Optional start date for filtering transactions
        end_date: Optional end date for filtering transactions
        **kwargs: Additional parameters (e.g., limit)
    
    Returns:
        tuple: (rfm_data, rfm_counts) - Customer segments and segment counts
    """
    
    # Extract data from dictionary
    states_df = data_dict.get('states')
    loyalty_df = data_dict.get('loyalty')
    transactions_df = data_dict.get('transactions')
    branch_df = data_dict.get('branches')
    
    # Validate required data
    if states_df is None or transactions_df is None:
        raise ValueError("'states' and 'transactions' DataFrames are required")
    
    if states_df.empty or transactions_df.empty:
        raise ValueError("Input DataFrames cannot be empty")
    
    logger.info(f"Processing {len(states_df)} customer records")
    
    # Apply date filters if provided
    if start_date or end_date:
        transactions_df = transactions_df.copy()
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        if start_date:
            transactions_df = transactions_df[transactions_df['date'] >= start_date]
        if end_date:
            transactions_df = transactions_df[transactions_df['date'] <= end_date]
        logger.info(f"Filtered to {len(transactions_df)} transactions")

    # Preprocess customer data
    logger.info("Preprocessing customer data")
    
    # Exclude test users and create clean copy
    state_df = states_df[~states_df['name'].str.startswith('Test User', na=False)].copy()

    # Capitalize each word in the 'name' column
    state_df['name'] = state_df['name'].str.title()

    # Initialize gender detector
    d = gender.Detector(case_sensitive=False)

    # Gender Prediction
    def predict_gender(name):
        """Predict gender from name using titles and name patterns."""
        if pd.isnull(name):
            return 'unknown'
        first = str(name).split()[0]
        titles_female = ['ms', 'mrs', 'miss']
        titles_male = ['mr']
        if first.lower() in titles_female:
            return 'female'
        if first.lower() in titles_male:
            return 'male'
        g = d.get_gender(first)
        if g in ['female', 'mostly_female']:
            return 'female'
        elif g in ['male', 'mostly_male']:
            return 'male'
        else:
            return 'unknown'

    state_df['predicted_gender'] = state_df['name'].apply(predict_gender)

    # Age Calculation
    def calculate_age(dob):
        """Calculate age from date of birth."""
        if pd.isnull(dob):
            return None
        try:
            birth_date = pd.to_datetime(dob, errors='coerce')
            if pd.isnull(birth_date):
                return None
            today = datetime.today()
            age = today.year - birth_date.year - (
                (today.month, today.day) < (birth_date.month, birth_date.day)
            )
            return age
        except Exception:
            return None
    
    state_df['age'] = state_df['dob'].apply(calculate_age)
    
    # Rename brand to branch_no if it exists
    if 'brand' in state_df.columns:
        state_df = state_df.rename(columns={'brand': 'branch_no'})

    # Merge loyalty accounts into state data
    # This adds accounts that exist in Loyalty but not in State
    if loyalty_df is not None and not loyalty_df.empty:
        loyalty_df = loyalty_df.rename(columns={'account_no': 'customer_id'})
        state_df = pd.concat(
            [state_df, loyalty_df[['customer_id', 'createdAt', 'updatedAt']]], 
            axis=0
        ).drop_duplicates(subset=['customer_id'], keep='first').reset_index(drop=True)
    
    # Process transaction data
    transactions_df = transactions_df.rename(columns={'loyalty_account': 'customer_id'})
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    transactions_df = transactions_df.drop_duplicates()
    
    # Filter to Sales and Refunds only
    transactions_df = transactions_df[
        transactions_df['trx_type'].isin(['Sale', 'Refund'])
    ].copy()
    
    # Calculate total sale value (quantity * unit price)
    transactions_df['sale_value'] = (
        transactions_df['sale_value'] * transactions_df['qty']
    ).astype(float)
    
    # Apply negative values for refunds
    transactions_df.loc[
        transactions_df['trx_type'] == 'Refund', 'sale_value'
    ] *= -1

    # Calculate Loyalty Period
    state_df['createdAt'] = pd.to_datetime(state_df['createdAt'], errors='coerce')
    state_df['updatedAt'] = pd.to_datetime(state_df['updatedAt'], errors='coerce')
    state_df['loyalty_days'] = (pd.Timestamp.today() - state_df['createdAt']).dt.days

    # Calculate RFM Metrics
    logger.info("Calculating RFM metrics")
    
    # Monetary: Total sales value per customer
    monetary = transactions_df.groupby('customer_id')['sale_value'].sum().reset_index()

    # Frequency: Number of unique transactions per customer
    frequency = (
        transactions_df.groupby('customer_id')['transaction_id']
        .nunique()
        .reset_index()
        .rename(columns={'transaction_id': 'frequency'})
    )

    # Recency: Days since last transaction
    recency = transactions_df.groupby('customer_id')['date'].max().reset_index()
    today = pd.Timestamp(datetime.today())
    recency['recency'] = (today - recency['date']).dt.days
    recency = recency.drop('date', axis=1)

    # Merge RFM metrics
    rfm = (
        recency
        .merge(frequency, on='customer_id', how='outer')
        .merge(monetary, on='customer_id', how='outer')
        .merge(state_df[['customer_id', 'loyalty_days']], on='customer_id', how='left')
    )
    
    # %% [markdown]
    # #### Creating Customers Clusters using KMeans

    # %%
    # Fill missing values with median for clustering,
    # including 'loyalty_days'

    rfm_filled = rfm[['recency', 'frequency', 'sale_value',
                    'loyalty_days']].fillna(
        rfm[['recency', 'frequency', 'sale_value', 
            'loyalty_days']].median()
    )

    # Standardize RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_filled)

    # KMeans clustering into 5 groups
    kmeans = KMeans(n_clusters=5, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Assign segment labels based on cluster centroids
    centroids = kmeans.cluster_centers_
    labels = ['High Value', 'Loyal', 'Potential', 'At Risk', 
            'Churn Risk']

    
    #Sort clusters by: low recency, high frequency, 
    #high monetary (sale_value), high loyalty_days
    
    cluster_order = sorted(
        range(5),
        key=lambda i: (centroids[i][0], -centroids[i][1], 
                    -centroids[i][2], -centroids[i][3])
    )
    segment_map = {cluster: labels[i] for cluster, 
                i in zip(cluster_order, range(5))}
    rfm['segment'] = rfm['cluster'].map(segment_map)

    # Show segment counts
    rfm_counts = rfm['segment'].value_counts().reset_index()
    rfm_counts['timestamp'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    rfm_counts['resolution'] = 'day'

    print(rfm_counts)

    # %%
    rfm_demo = pd.merge(rfm, state_df[['customer_id', 'predicted_gender']], on='customer_id', how='left')
    # rfm_state = pd.merge(rfm, state_df, on='customer_id', how='left')
    rfm_demo = rfm_demo.rename(columns = {'predicted_gender': 'Predicted Gender',
                                          'recency': 'Recency',
                                          'frequency': 'Frequency',
                                          'sale_value': 'Monetary',
                                          'segment': 'Segment'})
                                          
    


    # Create score columns based on hard-coded thresholds
    def create_rfm_scores(df):
        """
        Create RFM score columns based on hard-coded business rules
        """
        df_scored = df.copy()
        
        # Fill missing values
        df_scored['loyalty_days'] = df_scored['loyalty_days'].fillna(
            df_scored['loyalty_days'].median()
        )
        df_scored['recency'] = df_scored['recency'].fillna(0)
        df_scored['frequency'] = df_scored['frequency'].fillna(0)
        df_scored['sale_value'] = df_scored['sale_value'].fillna(0)
        
        # Recency Score: <=30 then 4, 31-90 then 3, 91-180 then 2, >180 then 1
        def recency_score(days):
            if days <= 30:
                return 4
            elif days <= 90:
                return 3
            elif days <= 180:
                return 2
            else:
                return 1
        
        # Frequency Score: >=4 then 4, 3 then 3, 2 then 2, <=1 then 1
        def frequency_score(freq):
            if freq >= 4:
                return 4
            elif freq == 3:
                return 3
            elif freq == 2:
                return 2
            else:
                return 1
        
        # Monetary Score: >=500 then 4, 200-499 then 3, 50-199 then 2, <50 then 1
        def monetary_score(value):
            if value >= 500:
                return 4
            elif value >= 200:
                return 3
            elif value >= 50:
                return 2
            else:
                return 1
        
        # Loyalty Score: >=200 then 4, 120-199 then 3, 60-119 then 2, <60 then 1
        def loyalty_score(days):
            if days >= 200:
                return 4
            elif days >= 120:
                return 3
            elif days >= 60:
                return 2
            else:
                return 1
        
        # Apply scoring functions
        df_scored['R_Score'] = df_scored['recency'].apply(recency_score)
        df_scored['F_Score'] = df_scored['frequency'].apply(frequency_score)
        df_scored['M_Score'] = df_scored['sale_value'].apply(monetary_score)
        df_scored['L_Score'] = df_scored['loyalty_days'].apply(loyalty_score)
        
        # Create combined RFM score
        df_scored['RFM_Score'] = (df_scored['R_Score'].astype(str) + 
                                df_scored['F_Score'].astype(str) + 
                                df_scored['M_Score'].astype(str))
        
        # Create total score
        df_scored['Total_Score'] = (df_scored['R_Score'] + df_scored['F_Score'] + 
                                df_scored['M_Score'] + df_scored['L_Score'])
        
        return df_scored

    # Apply scoring to RFM data
    rfm_scored = create_rfm_scores(rfm)
    rfm_scored['RFM_Score'] = rfm_scored['RFM_Score'].astype(int)


    # Create 4 customer segments based on RFM scores
    def create_four_segments(df):
        """
        Create 4 customer segments based on RFM scores:
        - High Value: Best customers (high recency, frequency, monetary)
        - Loyal: Regular customers with good engagement
        - At Risk: Customers who haven't purchased recently but have value
        - Lost: Customers with low engagement across all metrics
        """
        df_segmented = df.copy()
        
        def assign_segment(row):
            r_score = row['R_Score']
            f_score = row['F_Score'] 
            m_score = row['M_Score']
            l_score = row['L_Score']
            total = row['Total_Score']
            
            # High Value: Recent purchases, high frequency, high monetary value
            if (r_score >= 3 and f_score >= 3 and m_score >= 3) or total >= 15:
                return 'High Value'
            
            # Active: Good frequency or long loyalty, moderate recency/monetary
            elif (f_score >= 3 or l_score >= 3) and r_score >= 2:
                return 'Active'
            
            # Infrequent: Had value before but recent activity is low
            elif (m_score >= 2 or f_score >= 2) and r_score <= 2:
                return 'Infrequent'
            
            # Lost: Low scores across all metrics
            else:
                return 'Inactive'
        
        df_segmented['Segment'] = df_segmented.apply(assign_segment, axis=1)
        
        return df_segmented

    # Apply segmentation
    rfm_segmented = create_four_segments(rfm_scored)
    logger.info("Customer segmentation completed")

    # Display segment distribution
    segment_counts = rfm_segmented['Segment'].value_counts()
    segment_pct = (segment_counts / len(rfm_segmented) * 100).round(2)
    
    print("\nCustomer Segment Distribution:")
    for segment in ['High Value', 'Active', 'Infrequent', 'Inactive']:
        if segment in segment_counts:
            count = segment_counts[segment]
            pct = segment_pct[segment]
            print(f"{segment}: {count} ({pct}%)")

    # Merge with customer demographics
    demographic_cols = ['customer_id', 'name', 'email', 'phone', 'branch_no', 
                        'predicted_gender', 'age']
    available_cols = [col for col in demographic_cols if col in state_df.columns]
    rfm_demo = pd.merge(rfm_segmented, state_df[available_cols], on='customer_id', how='left')
    
    # Add branch information if available
    if branch_df is not None and not branch_df.empty:
        branch_df = branch_df.copy()
        branch_df['no'] = branch_df['no'].astype('str')
        
        # Merge branch names
        branch_mapping = branch_df[['no', 'name']].rename(
            columns={'no': 'branch_no', 'name': 'branch_name'}
        )
        rfm_demo = pd.merge(rfm_demo, branch_mapping, on='branch_no', how='left')
        
        # For customers with missing branch, find most frequent transaction branch
        if 'branch' in transactions_df.columns:
            missing_branch_customers = rfm_demo[
                rfm_demo['branch_name'].isna()
            ]['customer_id'].unique()
            
            if len(missing_branch_customers) > 0:
                frequent_branches = (
                    transactions_df[transactions_df['customer_id'].isin(missing_branch_customers)]
                    .groupby(['customer_id', 'branch'])
                    .size()
                    .reset_index(name='count')
                    .loc[lambda x: x.groupby('customer_id')['count'].idxmax()]
                    [['customer_id', 'branch']]
                )
                frequent_branches['branch'] = frequent_branches['branch'].astype('str')
                frequent_branches = frequent_branches.merge(
                    branch_df[['no', 'name']].rename(columns={'no': 'branch', 'name': 'branch_name_inferred'}),
                    on='branch',
                    how='left'
                )
                rfm_demo = rfm_demo.merge(frequent_branches[['customer_id', 'branch_name_inferred']], 
                                         on='customer_id', how='left')
                rfm_demo['branch_name'] = rfm_demo['branch_name'].fillna(
                    rfm_demo['branch_name_inferred']
                )
                rfm_demo = rfm_demo.drop('branch_name_inferred', axis=1)
    
    # Prepare final output
    rfm_counts = rfm_demo['Segment'].value_counts().reset_index()
    rfm_counts.columns = ['Segment', 'Count']
    rfm_counts['timestamp'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    # Return final results
    logger.info(f"Analysis completed successfully for {len(rfm_demo)} customers")
    return rfm_demo, rfm_counts


# Example usage
if __name__ == "__main__":
    # Sample data structure - replace with your actual data sources
    # Expected DataFrame schemas:
    # 
    # states_df columns: customer_id, name, dob, brand/branch_no, createdAt, updatedAt
    # loyalty_df columns: account_no, createdAt, updatedAt
    # line_df columns: loyalty_account, date, trx_type, sale_value, qty, transaction_id, branch
    # branch_df columns: no, name
    
    data_dict = {
        'states': pd.DataFrame(),      # Customer master data
        'loyalty': pd.DataFrame(),     # Loyalty program data
        'transactions': pd.DataFrame(), # Transaction line items
        'branches': pd.DataFrame()     # Branch/store information
    }
    
    # Run analysis
    rfm_results, segment_counts = execute_analysis(
        data_dict=data_dict,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    print("\nRFM Analysis Results:")
    print(rfm_results.head())
    print("\nSegment Distribution:")
    print(segment_counts)
