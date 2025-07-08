import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import datetime
import joblib # For saving the model
import os # Import os module

# --- Synthetic Data Generation ---
def generate_synthetic_data(num_samples=10000, fraud_rate=0.0179):
    np.random.seed(42) # for reproducibility

    data = {
        'transaction_id': range(num_samples),
        'amount': np.random.lognormal(mean=2.5, sigma=1.0, size=num_samples), # Skewed amounts
        'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_samples, freq='h').to_numpy() +
                               np.random.randint(-3600*24*30, 3600*24*30, num_samples).astype('timedelta64[s]')),
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

    # Introduce some fraud patterns
    num_fraud_samples = int(num_samples * fraud_rate)
    fraud_indices = np.random.choice(num_samples, num_fraud_samples, replace=False)
    df.loc[fraud_indices, 'is_fraud'] = 1

    high_amount_fraud_indices = df[df['amount'] > df['amount'].quantile(0.95)].sample(frac=0.3, replace=True).index
    df.loc[high_amount_fraud_indices, 'is_fraud'] = 1

    fraudulent_users = np.random.choice(df['user.id'].unique(), size=int(len(df['user.id'].unique()) * 0.01), replace=False)
    df.loc[df['user.id'].isin(fraudulent_users), 'is_fraud'] = np.random.choice([0, 1], p=[0.5, 0.5], size=df[df['user.id'].isin(fraudulent_users)].shape[0])

    actual_fraud_rate = df['is_fraud'].mean()
    print(f"Synthetic dataset created with {num_samples} samples. Fraud rate: {actual_fraud_rate:.2%}")

    return df

# --- Feature Engineering Function ---
def engineer_features(df):
    """
    Engineers new features from the raw transaction data.
    """
    print("Starting feature engineering...")

    df['date'] = pd.to_datetime(df['date'])
    df['__original_order__'] = df.index
    df = df.reset_index(drop=True)

    df['timestamp'] = df['date'].apply(lambda x: x.timestamp())
    df['hour_of_day'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day

    df_sorted_user = df.sort_values(by=['user.id', 'date']).copy()
    df_sorted_user['time_since_last_user_txn'] = df_sorted_user.groupby('user.id')['date'].diff().dt.total_seconds().fillna(0)
    df = df.merge(df_sorted_user[['__original_order__', 'time_since_last_user_txn']],
                  on='__original_order__', how='left')
    del df_sorted_user

    df_sorted_org = df.sort_values(by=['organization.id', 'date']).copy()
    df_sorted_org['time_since_last_org_txn'] = df_sorted_org.groupby('organization.id')['date'].diff().dt.total_seconds().fillna(0)
    df = df.merge(df_sorted_org[['__original_order__', 'time_since_last_org_txn']],
                  on='__original_order__', how='left')
    del df_sorted_org

    df_rolling_org = df.sort_values(by=['organization.id', 'date']).copy()
    df_rolling_org['temp_rolling_idx'] = df_rolling_org['date'] + pd.to_timedelta(df_rolling_org.groupby('organization.id').cumcount(), unit='ns')
    df_rolling_org = df_rolling_org.set_index('temp_rolling_idx')

    rolling_sum_org = df_rolling_org.groupby('organization.id')['amount'].rolling('24h', closed='left').sum().rename('rolling_sum_24hr_org')
    rolling_max_org = df_rolling_org.groupby('organization.id')['amount'].rolling('24h', closed='left').max().rename('rolling_max_24hr_org')

    results_org = pd.concat([rolling_sum_org, rolling_max_org], axis=1).reset_index(level=0, drop=True)
    results_org['__original_order__'] = df_rolling_org['__original_order__']
    results_org = results_org.reset_index(drop=True)

    df = df.merge(results_org, on='__original_order__', how='left')
    del df_rolling_org, results_org, rolling_sum_org, rolling_max_org


    df_rolling_user = df.sort_values(by=['user.id', 'date']).copy()
    df_rolling_user['temp_rolling_idx'] = df_rolling_user['date'] + pd.to_timedelta(df_rolling_user.groupby('user.id').cumcount(), unit='ns')
    df_rolling_user = df_rolling_user.set_index('temp_rolling_idx')

    rolling_avg_user = df_rolling_user.groupby('user.id')['amount'].rolling('24h', closed='left').mean().rename('rolling_avg_24hr_user')
    rolling_count_user = df_rolling_user.groupby('user.id')['transaction_id'].rolling('24h', closed='left').count().rename('rolling_count_24hr_user')

    results_user = pd.concat([rolling_avg_user, rolling_count_user], axis=1).reset_index(level=0, drop=True)
    results_user['__original_order__'] = df_rolling_user['__original_order__']
    results_user = results_user.reset_index(drop=True)

    df = df.merge(results_user, on='__original_order__', how='left')
    del df_rolling_user, results_user, rolling_avg_user, rolling_count_user


    for col in ['rolling_sum_24hr_org', 'rolling_avg_24hr_user', 'rolling_count_24hr_user', 'rolling_max_24hr_org',
                'time_since_last_user_txn', 'time_since_last_org_txn']:
        df[col] = df[col].fillna(0)


    df['user_avg_amount'] = df.groupby('user.id')['amount'].transform('mean')
    df['user_max_amount'] = df.groupby('user.id')['amount'].transform('max')

    df['org_avg_amount'] = df.groupby('organization.id')['amount'].transform('mean')
    df['org_max_amount'] = df.groupby('organization.id')['amount'].transform('max')

    df['amount_to_user_avg_ratio'] = df['amount'] / (df['user_avg_amount'] + 1e-6)

    df = df.sort_values(by='__original_order__').reset_index(drop=True)

    df = df.drop(columns=['date', 'transaction_id', '__original_order__'])

    print("Feature engineering complete.")
    return df

# --- Main Script Execution ---
if __name__ == "__main__":
    print("Loading/Generating synthetic public fraud data...")
    df = generate_synthetic_data(num_samples=10000, fraud_rate=0.0179)

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    X_engineered = engineer_features(X.copy())

    numerical_features = [
        'amount', 'timestamp', 'time_since_last_user_txn', 'time_since_last_org_txn',
        'rolling_sum_24hr_org', 'rolling_avg_24hr_user', 'rolling_count_24hr_user', 'rolling_max_24hr_org',
        'user_avg_amount', 'user_max_amount', 'org_avg_amount', 'org_max_amount',
        'amount_to_user_avg_ratio'
    ]
    categorical_features = [
        'user.id', 'organization.id', 'product.id', 'merchant.id',
        'transaction.type', 'device.type', 'location.country',
        'hour_of_day', 'day_of_week', 'month', 'day_of_month'
    ]

    engineered_cols = set(X_engineered.columns)
    defined_features = set(numerical_features + categorical_features)
    if engineered_cols != defined_features:
        print(f"Warning: Mismatch in engineered features and defined lists.")
        print(f"Missing from lists: {engineered_cols - defined_features}")
        print(f"Extra in lists: {defined_features - engineered_cols}")
        numerical_features = [col for col in numerical_features if col in engineered_cols]
        categorical_features = [col for col in categorical_features if col in engineered_cols]

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.3, random_state=42, stratify=y)

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
                                    ])

    print("\nTraining model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    print("\nEvaluating model...")
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # Save the trained model
    model_filename = 'fraud_detector_model.pkl'
    # Use os.path.abspath to get the full, absolute path
    save_path = os.path.abspath(model_filename)
    
    print(f"\nAttempting to save model to: {save_path}")

    try:
        joblib.dump(model_pipeline, save_path)
        print(f"Model saved successfully at '{save_path}'")
        # Add a check here if the file exists *immediately* after saving
        if os.path.exists(save_path):
            print(f"Confirmation: File '{save_path}' exists after saving.")
        else:
            print(f"WARNING: File '{save_path}' DOES NOT EXIST after saving. This is unexpected.")
    except Exception as e:
        print(f"ERROR: Failed to save model: {e}")