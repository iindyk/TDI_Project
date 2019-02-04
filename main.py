import bs4 as bs
import datetime as dt
import os
import requests
import pandas_datareader.data as web
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher
from scipy import stats


def similar(a, b):
    # measures similarity between strings
    return SequenceMatcher(None, a, b).ratio()


def save_sp500_tickers():
    # scrabble names of companies, their tickers and sectors from S&P 500 from Wikipedia
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        tickers.append({'ticker': row.findAll('td')[0].text,
                        'name': row.findAll('td')[1].text,
                        'sector': row.findAll('td')[3].text.replace('\n', '')})
    # save everything into pickle
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
        f.close()
    return tickers


def get_data_from_yahoo():
    # load stock prices data from yahoo
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
        f.close()
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker['ticker'])):
            df = web.DataReader(ticker['ticker'], 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker['ticker']))
        else:
            print('Already have {}'.format(ticker['ticker']))


def compile_data():
    # join stocks data into a single dataframe
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
        f.close()

    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker['ticker']))
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': ticker['ticker']}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('joined_closes_daily.csv')


def companies_bar_graph():
    n = 15
    first_n = dict(sorted(r_s.items(), key=lambda x: x[1], reverse=True)[:n])
    labels = first_n.keys()
    labels = [l.replace(' ', '\n').replace('&', '') for l in labels]
    vals = first_n.values()
    x = np.arange(n)
    fig, ax = plt.subplots()
    plt.bar(x, vals, width=0.4)
    plt.xticks(x, labels=labels)
    plt.ylabel('Coefficient of determination')
    plt.title('Dependence between monthly stock price and number of employees')
    plt.ylim(top=1., bottom=0.)
    plt.show()


def sectors_bar_graph():
    sectors = {}
    for name, rs in r_s.items():
        ticker = names_found[name]
        for t in tickers_all:
            if t['ticker'] == ticker:
                if t['sector'] in sectors:
                    sectors[t['sector']].append(rs)
                else:
                    sectors[t['sector']] = [rs]
    for s in sectors:
        sectors[s] = np.mean(np.array(sectors[s]))
    sectors = dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True))
    x = np.arange(len(sectors))
    fig, ax = plt.subplots()
    plt.bar(x, sectors.values(), width=0.4)
    plt.xticks(x, labels=sectors.keys())
    plt.ylabel('Average coefficient of determination')
    plt.title('Dependence between monthly stock price and number of employees by GICS sectors')
    plt.ylim(top=1., bottom=0.)
    plt.show()


if __name__ == '__main__':
    # open Linkedin and S&P 500 companies data
    data = pd.read_csv('temp_datalab_records_linkedin_company.csv')
    names_all = set(data['company_name'])
    with open("sp500tickers.pickle", "rb") as f:
        tickers_all = pickle.load(f)
        f.close()

    # match company names by similarity
    names_found = {}    # dictionary {name: ticker}
    for ticker in tickers_all:
        for name in names_all:
            if similar(ticker['name'], name) > 0.85:
                names_found[name] = ticker['ticker']
                break

    # read prices data
    prices = pd.read_csv('joined_closes_daily.csv')
    r_s = {}

    # match prices and employees data, calculate r**2
    for name, ticker in names_found.items():
        x_daily = data[data['company_name'] == name][['as_of_date', 'employees_on_platform']]
        x_daily.as_of_date = pd.to_datetime(x_daily.as_of_date)
        x_monthly = x_daily.resample('M', on='as_of_date', how={'as_of_date': 'last', 'employees_on_platform': 'last'})
        y_daily = prices[['Date', ticker]]
        y_daily.Date = pd.to_datetime(y_daily.Date)
        y_monthly = y_daily.resample('M', on='Date', how={ticker: 'last', 'Date': 'last'})
        if len(x_monthly) > 12:
            joined_xy = pd.merge(x_monthly, y_monthly, how='inner', left_on='as_of_date', right_on='Date',
                                 left_index=True, right_index=True)
            _, _, r_value, _, _ = stats.linregress(np.array(joined_xy['employees_on_platform']),
                                                   np.array(joined_xy[ticker]))
            if not np.isnan(r_value):
                r_s[name] = r_value**2

    sectors_bar_graph()


