import joblib
import pandas as pd

bundle_path = 'models/bundle.joblib'
bundle = joblib.load(bundle_path)
portfolio = bundle['portfolio']

# Save to CSV
portfolio.to_csv('moj_portfel_300_firm.csv', index=False, encoding='utf-8')
print('moj_portfel_300_firm.csv generated successfully.')
