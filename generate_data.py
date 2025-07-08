# generate_data.py
import pandas as pd
import numpy as np
import datetime

def generate_synthetic_data(num_samples=10000, fraud_rate=0.0179):
    """Generates synthetic transaction data."""
    np.random.seed(42) # for reproducibility

    # Define a start date for the synthetic transactions (e.g., last 365 days from now)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    time_delta_seconds = (end_date - start_date).total_seconds()

    data = {
        'transaction_id': np.arange(num_samples),
        'amount': np.random.lognormal(mean=2.5, sigma=1.0, size=num_samples), # Skewed amounts
        # Generate dates within the last year, randomly
        'date': [start_date + datetime.timedelta(seconds=np.random.uniform(0, time_delta_seconds)) for _ in range(num_samples)],
        'user.id': np.random.randint(1, num_samples // 5, num_samples),
        'organization.id': np.random.randint(1, num_samples // 10, num_samples),
        'product.id': np.random.randint(1, 20, num_samples),
        'merchant.id': np.random.randint(1, num_samples // 8, num_samples),
        'transaction.type': np.random.choice(['purchase', 'refund', 'withdrawal', 'deposit'], num_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'device.type': np.random.choice(['mobile', 'desktop', 'tablet'], num_samples, p=[0.6, 0.3, 0.1]),
        'location.country': np.random.choice(['USA', 'CAN', 'GBR', 'AUS', 'DEU', 'FRA'], num_samples, p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1]),
        'is_fraud': np.zeros(num_samples, dtype=int)
    }

    df = pd.DataFrame(data)

    # Introduce some fraud patterns similar to the original synthetic data generation
    num_fraud_samples = int(num_samples * fraud_rate)
    fraud_indices = np.random.choice(num_samples, num_fraud_samples, replace=False)
    df.loc[fraud_indices, 'is_fraud'] = 1

    high_amount_fraud_indices = df[df['amount'] > df['amount'].quantile(0.95)].sample(frac=0.3, replace=True, random_state=42).index
    df.loc[high_amount_fraud_indices, 'is_fraud'] = 1

    fraudulent_users = np.random.choice(df['user.id'].unique(), size=int(len(df['user.id'].unique()) * 0.01), replace=False)
    df.loc[df['user.id'].isin(fraudulent_users), 'is_fraud'] = np.random.choice([0, 1], p=[0.5, 0.5], size=df[df['user.id'].isin(fraudulent_users)].shape[0])


    df.sort_values(by='date', inplace=True) # Sort by date for time-based features
    df = df.reset_index(drop=True) # Reset index after sorting

    return df

if __name__ == '__main__':
    # Example usage:
    synthetic_df = generate_synthetic_data(num_samples=1000)
    print(synthetic_df.head())
    print(f"Total synthetic transactions: {len(synthetic_df)}")
    print(f"Total synthetic fraud: {synthetic_df['is_fraud'].sum()}")