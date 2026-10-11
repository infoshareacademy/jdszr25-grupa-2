import joblib
import pandas as pd
import numpy as np

bundle_path = 'models/bundle.joblib'
bundle = joblib.load(bundle_path)

# Load real company names
df = pd.read_parquet('C:/Users/j.zawadzki/Desktop/AI- data szkolenie/projekt/jdszr25-grupa-2-main/V4FinBench_data_load-main/processed/company_years_h1.parquet', columns=['company'])
companies = df['company'].dropna().unique().tolist()

# Get the current portfolio
portfolio = bundle['portfolio']
n_firms = len(portfolio)

# Pick random unique real companies
np.random.seed(42)
real_names = np.random.choice(companies, n_firms, replace=False)

portfolio['firma'] = real_names
bundle['portfolio'] = portfolio

joblib.dump(bundle, bundle_path)
print('Portfolio updated with real company names!')
