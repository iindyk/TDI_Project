import pandas as pd
import pickle
import numpy as np
from difflib import SequenceMatcher
from scipy import stats


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


data = pd.read_csv('temp_datalab_records_linkedin_company.csv')
# 'Financial Services', 'Investment Banking', 'Banking'
names_all = set(data['company_name'])
with open("sp500tickers.pickle", "rb") as f:
    tickers_all = pickle.load(f)
    f.close()

names_found = {}    # dictionary {name: ticker}
for ticker in tickers_all:
    for name in names_all:
        if similar(ticker['name'], name) > 0.85:
            names_found[name] = ticker['ticker']
            break

prices = pd.read_csv('joined_closes_daily.csv')

r_s = {}
for name, ticker in names_found.items():
    x_daily = data[data['company_name'] == name][['as_of_date', 'employees_on_platform']]
    x_daily.as_of_date = pd.to_datetime(x_daily.as_of_date)
    x_monthly = x_daily.resample('M', on='as_of_date', how={'as_of_date': 'last', 'employees_on_platform': 'last'})
    y_daily = prices[['Date', ticker]]
    y_daily.Date = pd.to_datetime(y_daily.Date)
    y_monthly = y_daily.resample('M', on='Date', how={ticker: 'last', 'Date': 'last'})

    joined_xy = pd.merge(x_monthly, y_monthly, how='inner', left_on='as_of_date', right_on='Date',
                         left_index=True, right_index=True)
    _, _, r_value, _, _ = stats.linregress(np.array(joined_xy['employees_on_platform']),
                                           np.array(joined_xy[ticker]))
    r_s[name] = r_value**2

print(r_s)


