import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
import datetime
import joblib
import os
import json
import requests # Import the requests library for API calls
from dotenv import load_dotenv # Recommended for managing environment variables

# Load environment variables from .env file (if present)
load_dotenv()

# --- Flask App Setup ---
app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'fraud_detector_model.pkl'

# --- HCB API Configuration ---
HCB_API_BASE_URL = "https://hcb.hackclub.com/api/v3"
# IMPORTANT: Get your HCB API key if you need to access non-public data.
# It's HIGHLY recommended to store this in an environment variable.
# For example, in a .env file: HCB_API_KEY="sk_live_..."
HCB_API_KEY = os.getenv("HCB_API_KEY") # No default here, rely on .env or actual env var

# --- Data Fetching from HCB API ---
def fetch_hcb_transactions(organization_id=None, limit=50000):
    """
    Fetches transaction data from the HCB API.
    organization_id: int, optional. Filters transactions for a specific organization.
                     If None, will return an empty DataFrame as global transactions are not directly available.
    limit: int. Maximum number of transactions to fetch.
    """
    transactions_data = []
    headers = {}
    if HCB_API_KEY:
        headers = {"Authorization": f"Bearer {HCB_API_KEY}"}
        print("DEBUG: Using HCB API Key for authentication.")
    else:
        print("WARNING: HCB_API_KEY environment variable is not set. Access might be limited to public data.")

    # --- IMPORTANT FIX HERE ---
    if organization_id is None:
        print("INFO: Cannot fetch general transactions globally from HCB API directly without an organization ID.")
        print("INFO: Returning empty data for this request. Consider fetching transparent organizations first for a 'global' view.")
        return pd.DataFrame() # Return an empty DataFrame directly
    # --- END IMPORTANT FIX ---

    # If organization_id is provided, proceed to fetch specific organization's transactions
    url = f"{HCB_API_BASE_URL}/organizations/{organization_id}/transactions"
    params = {"per_page": 100} # Fetch 100 items per page

    print(f"INFO: Fetching transactions for organization ID: {organization_id} from {url}")

    page = 1
    fetched_count = 0

    while fetched_count < limit:
        current_params = params.copy()
        current_params["page"] = page

        try:
            response = requests.get(url, headers=headers, params=current_params, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            current_page_data = response.json()

            # Defensive: If the response is not a list, it's likely an error (e.g., {"message": ...})
            if not isinstance(current_page_data, list):
                print(f"WARNING: HCB API did not return a list for org {organization_id}. Response: {current_page_data}")
                return pd.DataFrame() # Return empty DataFrame

            if not current_page_data:
                break # No more data

            transactions_data.extend(current_page_data)
            fetched_count += len(current_page_data)
            page += 1

            if len(current_page_data) < params["per_page"]:
                break # Last page, no need to fetch more

            print(f"DEBUG: Fetched {fetched_count} transactions so far...")

        except requests.exceptions.HTTPError as http_err:
            print(f"ERROR: HTTP error occurred while fetching HCB transactions for org {organization_id}: {http_err} - Response: {getattr(http_err.response, 'text', '')}")
            return pd.DataFrame() # Return empty DataFrame on HTTP error
        except requests.exceptions.ConnectionError as conn_err:
            print(f"ERROR: Connection error while fetching HCB transactions for org {organization_id}: {conn_err}")
            return pd.DataFrame()
        except requests.exceptions.Timeout as timeout_err:
            print(f"ERROR: Timeout error while fetching HCB transactions for org {organization_id}: {timeout_err}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as req_err:
            print(f"ERROR: An unexpected request error occurred while fetching HCB transactions for org {organization_id}: {req_err}")
            return pd.DataFrame()
        except json.JSONDecodeError as json_err:
            print(f"ERROR: Failed to decode JSON response from HCB API for org {organization_id}: {json_err} - Response text: {response.text}")
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR: An unexpected error occurred in HCB fetch for org {organization_id}: {e}")
            return pd.DataFrame()

    if not transactions_data:
        print(f"WARNING: No transactions data fetched for org {organization_id} from HCB API.")
        return pd.DataFrame()

    # Map HCB API fields to our expected DataFrame columns
    mapped_data = []
    for txn in transactions_data:
        # HCB amount is typically in cents, convert to dollars
        amount_usd = float(txn.get('amount', 0)) / 100.0 if txn.get('amount') is not None else 0.0

        mapped_data.append({
            'transaction_id': txn.get('id'),
            'amount': amount_usd,
            'date': txn.get('date'), # HCB date is 'YYYY-MM-DD'
            'user.id': txn.get('creator', {}).get('id', np.random.randint(1, 1000)), # Using creator ID or random
            'organization.id': txn.get('organization', {}).get('id', 0), # Organization ID is crucial
            'product.id': np.random.randint(1, 20), # Placeholder: HCB API usually doesn't have a 'product.id'
            'merchant.id': txn.get('merchant', {}).get('id', np.random.randint(1, 1000)), # Using merchant ID or random
            'transaction.type': txn.get('type', 'unknown'), # e.g., 'withdrawal', 'deposit', 'invoice'
            'device.type': np.random.choice(['mobile', 'desktop', 'tablet']), # Placeholder: HCB API doesn't have 'device.type'
            'location.country': np.random.choice(['USA', 'CAN', 'GBR', 'AUS', 'DEU', 'FRA']), # Placeholder: HCB API transactions don't expose location country easily
            'is_fraud': 0 # This is the target variable; real API data won't have it. Assume non-fraud for raw data.
        })

    df = pd.DataFrame(mapped_data)

    if df.empty:
        print(f"WARNING: DataFrame is empty after mapping for org {organization_id}.")
        return df

    # Basic cleaning after mapping
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['transaction_id', 'amount', 'date', 'organization.id'], inplace=True) # Drop rows with essential missing data

    print(f"INFO: Fetched and mapped {len(df)} transactions for org {organization_id} from HCB API.")
    return df

# --- Feature Engineering Function (MUST BE IDENTICAL TO train_model.py's version) ---
def engineer_features(df):
    """
    Engineers new features from the raw transaction data.
    IMPORTANT: This function will keep '__original_order__' and 'transaction_id'
               so they can be used for merging and display later.
    """
    if 'amount' not in df.columns:
        raise KeyError("EngineerFeaturesError: 'amount' column not found in input DataFrame.")
    if df.empty:
        print("WARNING: Empty DataFrame passed to engineer_features. Returning empty engineered DF.")
        all_possible_engineered_cols = [
            'transaction_id', 'amount', 'user.id', 'organization.id', 'product.id',
            'merchant.id', 'transaction.type', 'device.type', 'location.country',
            'is_fraud', '__original_order__', 'timestamp', 'hour_of_day',
            'day_of_week', 'month', 'day_of_month', 'time_since_last_user_txn',
            'time_since_last_org_txn', 'rolling_sum_24hr_org', 'rolling_max_24hr_org',
            'rolling_avg_24hr_user', 'rolling_count_24hr_user', 'user_avg_amount',
            'user_max_amount', 'org_avg_amount', 'org_max_amount', 'amount_to_user_avg_ratio'
        ]
        return pd.DataFrame(columns=all_possible_engineered_cols)


    df_copy = df.copy() # Work on a copy

    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy['__original_order__'] = df_copy.index
    df_copy = df_copy.reset_index(drop=True)

    df_copy['timestamp'] = df_copy['date'].apply(lambda x: x.timestamp())
    df_copy['hour_of_day'] = df_copy['date'].dt.hour
    df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['day_of_month'] = df_copy['date'].dt.day

    df_sorted_user = df_copy.sort_values(by=['user.id', 'date']).copy()
    df_sorted_user['time_since_last_user_txn'] = df_sorted_user.groupby('user.id')['date'].diff().dt.total_seconds().fillna(0)
    df_copy = df_copy.merge(df_sorted_user[['__original_order__', 'time_since_last_user_txn']],
                  on='__original_order__', how='left')
    del df_sorted_user

    df_sorted_org = df_copy.sort_values(by=['organization.id', 'date']).copy()
    df_sorted_org['time_since_last_org_txn'] = df_sorted_org.groupby('organization.id')['date'].diff().dt.total_seconds().fillna(0)
    df_copy = df_copy.merge(df_sorted_org[['__original_order__', 'time_since_last_org_txn']],
                  on='__original_order__', how='left')
    del df_sorted_org

    df_rolling_org = df_copy.sort_values(by=['organization.id', 'date']).copy()
    df_rolling_org['temp_rolling_idx'] = df_rolling_org['date'] + pd.to_timedelta(df_rolling_org.groupby('organization.id').cumcount(), unit='ns')
    df_rolling_org = df_rolling_org.set_index('temp_rolling_idx')

    df_rolling_org['amount'] = pd.to_numeric(df_rolling_org['amount'], errors='coerce')

    rolling_sum_org = df_rolling_org.groupby('organization.id')['amount'].rolling('24h', closed='left').sum().rename('rolling_sum_24hr_org')
    rolling_max_org = df_rolling_org.groupby('organization.id')['amount'].rolling('24h', closed='left').max().rename('rolling_max_24hr_org')

    results_org = pd.concat([rolling_sum_org, rolling_max_org], axis=1).reset_index(level=0, drop=True)
    results_org['__original_order__'] = df_rolling_org['__original_order__']
    results_org = results_org.reset_index(drop=True)

    df_copy = df_copy.merge(results_org, on='__original_order__', how='left')
    del df_rolling_org, results_org, rolling_sum_org, rolling_max_org


    df_rolling_user = df_copy.sort_values(by=['user.id', 'date']).copy()
    df_rolling_user['temp_rolling_idx'] = df_rolling_user['date'] + pd.to_timedelta(df_rolling_user.groupby('user.id').cumcount(), unit='ns')
    df_rolling_user = df_rolling_user.set_index('temp_rolling_idx')
    
    df_rolling_user['amount'] = pd.to_numeric(df_rolling_user['amount'], errors='coerce')

    rolling_avg_user = df_rolling_user.groupby('user.id')['amount'].rolling('24h', closed='left').mean().rename('rolling_avg_24hr_user')
    rolling_count_user = df_rolling_user.groupby('user.id')['transaction_id'].rolling('24h', closed='left').count().rename('rolling_count_24hr_user')

    results_user = pd.concat([rolling_avg_user, rolling_count_user], axis=1).reset_index(level=0, drop=True)
    results_user['__original_order__'] = df_rolling_user['__original_order__']
    results_user = results_user.reset_index(drop=True)

    df_copy = df_copy.merge(results_user, on='__original_order__', how='left')
    del df_rolling_user, results_user, rolling_avg_user, rolling_count_user


    for col in ['rolling_sum_24hr_org', 'rolling_avg_24hr_user', 'rolling_count_24hr_user', 'rolling_max_24hr_org',
                'time_since_last_user_txn', 'time_since_last_org_txn']:
        df_copy[col] = df_copy[col].fillna(0)


    df_copy['amount'] = pd.to_numeric(df_copy['amount'], errors='coerce')
    
    df_copy['user_avg_amount'] = df_copy.groupby('user.id')['amount'].transform('mean')
    df_copy['user_max_amount'] = df_copy.groupby('user.id')['amount'].transform('max')

    df_copy['org_avg_amount'] = df_copy.groupby('organization.id')['amount'].transform('mean')
    df_copy['org_max_amount'] = df_copy.groupby('organization.id')['amount'].transform('max')

    df_copy['amount_to_user_avg_ratio'] = df_copy['amount'] / (df_copy['user_avg_amount'].replace(0, np.nan) + 1e-6)
    df_copy['amount_to_user_avg_ratio'] = df_copy['amount_to_user_avg_ratio'].fillna(0)


    df_copy = df_copy.sort_values(by='__original_order__').reset_index(drop=True)

    df_copy = df_copy.drop(columns=['date'])

    return df_copy


# --- DataProcessor Class ---
class DataProcessor:
    def __init__(self, data):
        self.raw_data = data.copy()

        print(f"DEBUG: DataProcessor.__init__ received raw_data columns: {self.raw_data.columns.tolist()}")
        print(f"DEBUG: DataProcessor.__init__ received raw_data shape: {self.raw_data.shape}")

        self.processed_df = self._create_dataframe()

    def _create_dataframe(self):
        df = self.raw_data.copy()
        print(f"DEBUG: _create_dataframe input df columns (copy of raw_data): {df.columns.tolist()}")
        print(f"DEBUG: _create_dataframe input df shape: {df.shape}")

        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        required_cols_for_engineering = [
            'transaction_id', 'amount', 'date', 'user.id', 'organization.id',
            'product.id', 'merchant.id', 'transaction.type', 'device.type',
            'location.country', 'is_fraud'
        ]
        
        for col in required_cols_for_engineering:
            if col not in df.columns:
                print(f"WARNING: Column '{col}' not found in raw data for engineer_features. Adding with default.")
                if col == 'amount': df[col] = 0.0
                elif col == 'date': df[col] = pd.NaT
                elif col == 'is_fraud': df[col] = 0
                elif 'id' in col: df[col] = -1
                else: df[col] = 'unknown'

        df.dropna(subset=['amount', 'date', 'transaction_id', 'organization.id'], inplace=True)
        print(f"DEBUG: df columns after initial cleaning and ensuring all cols for engineering: {df.columns.tolist()}")
        print(f"DEBUG: df shape after initial cleaning and ensuring all cols for engineering: {df.shape}")


        if df.empty:
            print("WARNING: DataFrame became empty after initial cleaning steps. engineer_features will receive an empty DataFrame.")
            return engineer_features(pd.DataFrame(columns=df.columns))

        df_engineered = engineer_features(df.copy())
        print(f"DEBUG: _create_dataframe received df_engineered columns from engineer_features: {df_engineered.columns.tolist()}")
        print(f"DEBUG: _create_dataframe received df_engineered shape: {df_engineered.shape}")
        return df_engineered

    def get_dashboard_data(self):
        temp_raw_data_for_merge = self.raw_data.copy()
        temp_raw_data_for_merge['__original_order__'] = temp_raw_data_for_merge.index

        final_display_df = pd.merge(temp_raw_data_for_merge,
                                    self.processed_df,
                                    on='__original_order__',
                                    how='left',
                                    suffixes=('_raw', '_engineered'))

        core_display_cols = [
            'transaction_id', 'amount', 'date', 'user.id', 'organization.id',
            'product.id', 'merchant.id', 'transaction.type', 'device.type',
            'location.country', 'is_fraud'
        ]

        columns_to_drop = []
        columns_to_rename = {}

        for col in final_display_df.columns:
            if col.endswith('_raw') and col[:-4] in core_display_cols:
                columns_to_rename[col] = col[:-4]
            elif col.endswith('_engineered') and col[:-11] in core_display_cols:
                columns_to_drop.append(col)

        final_display_df.drop(columns=columns_to_drop, inplace=True)
        final_display_df.rename(columns=columns_to_rename, inplace=True)


        if '__original_order__' in final_display_df.columns:
            final_display_df.drop(columns=['__original_order__'], inplace=True)

        total_transactions = len(final_display_df)
        total_fraud = final_display_df['is_fraud'].sum() if 'is_fraud' in final_display_df.columns else 0
        fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0

        # --- FIX FOR daily_volume ---
        if 'date' in final_display_df.columns:
            final_display_df['date'] = pd.to_datetime(final_display_df['date'])
            daily_volume_df = final_display_df.groupby(final_display_df['date'].dt.date)['amount'].sum().reset_index()
            daily_volume_df['date'] = daily_volume_df['date'].astype(str)
            daily_volume = daily_volume_df.to_dict(orient='records') # Convert to list of dicts
        else:
            print("WARNING: 'date' column not found in final_display_df. Cannot calculate daily volume.")
            daily_volume = pd.DataFrame(columns=['date', 'amount']).to_dict(orient='records')

        # --- FIX FOR fraud_by_type ---
        if 'transaction.type' in final_display_df.columns and 'is_fraud' in final_display_df.columns:
            fraud_by_type_df = final_display_df.groupby('transaction.type')['is_fraud'].sum().reset_index()
            fraud_by_type = fraud_by_type_df.to_dict(orient='records') # Convert to list of dicts
        else:
            print("WARNING: 'transaction.type' or 'is_fraud' not found for fraud_by_type. Skipping.")
            fraud_by_type = pd.DataFrame(columns=['transaction.type', 'is_fraud']).to_dict(orient='records')


        if 'date' in final_display_df.columns:
            latest_transactions_df = final_display_df.sort_values(by='date', ascending=False).head(10).copy()
        else:
            latest_transactions_df = final_display_df.head(10).copy()


        ml_predictions_available = False
        if ml_model:
            ml_predictions_available = True
            
            raw_cols_for_engineer_input = [
                'transaction_id', 'amount', 'date', 'user.id', 'organization.id',
                'product.id', 'merchant.id', 'transaction.type', 'device.type',
                'location.country', 'is_fraud'
            ]
            
            temp_raw_for_latest_pred = latest_transactions_df[[col for col in raw_cols_for_engineer_input if col in latest_transactions_df.columns]].copy()
            
            if temp_raw_for_latest_pred.empty:
                ml_predictions_available = False
            else:
                if 'date' in temp_raw_for_latest_pred.columns:
                    temp_raw_for_latest_pred['date'] = pd.to_datetime(temp_raw_for_latest_pred['date'])
                
                engineered_latest_for_pred = engineer_features(temp_raw_for_latest_pred)
                
                if engineered_latest_for_pred.empty:
                    ml_predictions_available = False
                else:
                    cols_to_drop_for_model_input = ['transaction_id', '__original_order__', 'is_fraud']
                    
                    model_input_df = engineered_latest_for_pred.drop(
                        columns=[col for col in cols_to_drop_for_model_input if col in engineered_latest_for_pred.columns],
                        errors='ignore'
                    )
                    
                    try:
                        predicted_probas = ml_model.predict_proba(model_input_df)[:, 1]
                        latest_transactions_df['fraud_probability'] = predicted_probas
                    except Exception as ml_e:
                        print(f"ERROR: ML Model Prediction failed: {ml_e}")
                        ml_predictions_available = False

        latest_transactions = latest_transactions_df.to_dict(orient='records')
        for txn in latest_transactions:
            if 'date' in txn and isinstance(txn['date'], pd.Timestamp):
                txn['date'] = txn['date'].isoformat()

        return {
            'summary': {
                'total_transactions': total_transactions,
                'total_fraud': int(total_fraud),
                'fraud_rate': round(fraud_rate, 2)
            },
            'daily_volume': daily_volume, # This is now a list of dicts
            'fraud_by_type': fraud_by_type, # This is now a list of dicts
            'latest_transactions': latest_transactions,
            'ml_predictions': {
                'available': ml_predictions_available,
                'top_risks': []
            }
        }

# --- Load ML Model ---
ml_model = None
try:
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
        print(f"ML model '{MODEL_PATH}' loaded successfully.")
    else:
        print(f"Warning: ML model file '{MODEL_PATH}' not found. Run 'train_model.py' first to train the model. ML predictions will not be available.")
except Exception as e:
    print(f"Error loading ML model: {e}")
    ml_model = None

# --- Fetch all available organizations from HCB API ---
available_orgs = {}
print("Fetching available HCB organizations...")
try:
    orgs_url = f"{HCB_API_BASE_URL}/organizations"
    params = {"per_page": 100}
    response = requests.get(orgs_url, params=params, timeout=10)
    response.raise_for_status()
    orgs = response.json()
    if isinstance(orgs, list):
        print("\n--- Available HCB Organizations ---")
        for org in orgs:
            slug = org.get('slug', '').lower()
            org_id = org.get('id')
            name = org.get('name', 'Unknown')
            if slug and org_id:
                available_orgs[slug] = org_id
                print(f"  {slug} (ID: {org_id}) - {name}")
        print(f"--- Found {len(available_orgs)} organizations ---\n")
    else:
        print(f"Unexpected orgs response: {orgs}")
except Exception as e:
    print(f"Failed to fetch organizations: {e}")

# Use synthetic data for the global dashboard
print("Using synthetic data for global dashboard...")
from generate_data import generate_synthetic_data
all_transactions_df = generate_synthetic_data(num_samples=10000, fraud_rate=0.0179)
print("DEBUG: Using synthetic data as global fallback.")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/dashboard/hq', methods=['GET'])
def get_dashboard_data_route():
    try:
        processor = DataProcessor(all_transactions_df.copy())
        dashboard_data = processor.get_dashboard_data()
        dashboard_data['organization_context'] = 'Global (Synthetic)'
        return jsonify(dashboard_data)
    except (KeyError, ValueError) as e:
        print(f"Key/Value Error in get_dashboard_data_route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to load dashboard data: {e}. Check data source and feature engineering."}), 500
    except Exception as e:
        print(f"An unexpected error occurred in get_dashboard_data_route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/organizations', methods=['GET'])
def get_available_organizations():
    """Return list of all available HCB organizations"""
    org_list = []
    for slug, org_id in available_orgs.items():
        org_list.append({
            'slug': slug,
            'id': org_id,
            'name': slug.replace('-', ' ').title()
        })
    return jsonify({
        'organizations': org_list,
        'total': len(org_list)
    })

@app.route('/api/dashboard/<organization_name>', methods=['GET'])
def get_organization_dashboard_data(organization_name):
    print(f"Received request for organization: {organization_name}")
    
    # Check if organization exists in our available orgs
    target_org_id = available_orgs.get(organization_name.lower())
    
    if target_org_id is not None:
        print(f"INFO: Fetching real-time data for organization '{organization_name}' (ID: {target_org_id})...")
        # Fetch real data from HCB API
        filtered_data = fetch_hcb_transactions(organization_id=target_org_id, limit=5000)
        
        if not filtered_data.empty:
            try:
                processor = DataProcessor(filtered_data)
                dashboard_data = processor.get_dashboard_data()
                dashboard_data['organization_context'] = organization_name
                return jsonify(dashboard_data)
            except (KeyError, ValueError) as e:
                print(f"Key/Value Error for org dashboard: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Failed to load dashboard for {organization_name}: {e}"}), 500
            except Exception as e:
                print(f"An unexpected error occurred for org dashboard: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": "An internal server error occurred for organization dashboard."}), 500
        else:
            # If no real data, fall back to synthetic data filtered by a pseudo org ID
            print(f"INFO: No real data found for '{organization_name}'. Using synthetic data fallback.")
            synthetic_data = generate_synthetic_data(num_samples=5000, fraud_rate=0.0179)
            # Filter synthetic data to simulate organization-specific data
            org_filtered_data = synthetic_data[synthetic_data['organization.id'] == (target_org_id % 100)].copy()
            if org_filtered_data.empty:
                org_filtered_data = synthetic_data.head(100).copy()  # Fallback to some data
            
            try:
                processor = DataProcessor(org_filtered_data)
                dashboard_data = processor.get_dashboard_data()
                dashboard_data['organization_context'] = f"{organization_name} (Synthetic)"
                return jsonify(dashboard_data)
            except Exception as e:
                print(f"Error processing synthetic data for {organization_name}: {e}")
                return jsonify({"error": f"Failed to process data for {organization_name}"}), 500
    else:
        return jsonify({
            "message": f"Organization '{organization_name}' not found.", 
            "available_organizations": list(available_orgs.keys())
        }), 404


if __name__ == '__main__':
    app.run(debug=True)
