import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller  # Used to check for stationarity. Will be added again soon.

from pandas_datareader.data import DataReader

SEED = 1234
np.random.seed(SEED)

EARLIEST_DATE_BRENT_STR = '1987-05-20'
LAST_DATE_STR = '2022-10-1'

"""
References

[1] U.S. Energy Information Administration, Crude Oil Prices: 
West Texas Intermediate (WTI) - Cushing, Oklahoma [DCOILWTICO], 
retrieved from FRED, Federal Reserve Bank of St. Louis;
https://fred.stlouisfed.org/series/DCOILWTICO, June 8, 2023.
"""


# Read data and obtain recession labels
data_path = "data/DCOILWTICO_MAX.csv"
data = pd.read_csv(data_path)


# Earliest Brent Price is 1987-05-20 in dataset [1], so get US Recession data for only this range
earliest_date_brent_tup_int = (int(v) for v in EARLIEST_DATE_BRENT_STR.split('-'))
last_date_tup_int = (int(v) for v in LAST_DATE_STR.split('-'))

# Recession labels from National Bureau of Economic Research (NBER)
usrec = DataReader(
    "USREC", "fred",
    start=datetime(*tuple(earliest_date_brent_tup_int)),
    end=datetime(*tuple(last_date_tup_int))
)

# Earliest Brent Price is 1987-05-20 in dataset [1], so drop earlier WTI prices
drop_idx_below_val = (data[data['DATE'] == EARLIEST_DATE_BRENT_STR]).index[0]
data = data[drop_idx_below_val:].reset_index(drop=True)

"""
This outputs raw data and save copy that keeps DATE variable as variable, so we
can output this without issue later with resampled values and corresponding
task labels.
"""
data_dir = "data"
output_filename = "wti_prices_daily_raw.csv"
output_filepath = os.path.join(data_dir, output_filename)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
data.to_csv(output_filepath)

data_copy = data.copy()

"""
This converts dates into datetime and reindexes the data DataFrame so that
dates are indices
"""
data['DATE_DT'] = pd.to_datetime(data['DATE'])
data['DCOILWTICO'] = pd.to_numeric(data['DCOILWTICO'], errors='coerce')
data = data.set_index('DATE_DT')
data = data.drop('DATE', axis=1)

# TODO (Pranay): Store events_dict as csv or json and read in from file
# events_dict = {
#     "Q1 1986":"Saudis abandon swing producer role",
#     "Q2 1990":"Trough price prior to Iraq's invasion of Kuwait",
#     "Q3 1990":"Iraq invades Kuwait",
#     "Q4 1990":"Peak price during invasion",
#     "Q2 1991":"Iraq accepts UN resolution to end conflict",
#     "Q4 1996":"Peak price prior to Asian financial crisis",
#     "Q3 1997":"Asian financial crisis begins",
#     "Q1 1999":"OPEC cuts production target by 1.7M b/d",
#     "Q4 2000":"Peak price prior to 9/11",
#     "Q3 2001":"9/11 attacks",
#     "Q4 2001":"Trough price after 9/11",
#     "Q1 2005":"Low spare capacity",
#     "Q2 2008":"Peak price before global financial collapse",
#     "Q1 2009":"OPEC cuts production targets by 4.2M b/d",
#     "Q2 2014":"Peak price prior to supply gut price collapse",
#     "Q1 2015":"OPEC production quota unchanged despite low prices",
#     "Q4 2019":"Price immediately prior to global pandemic",
#     "Q1 2020":"COVID-19 declared a pandemic",
#     "Q2 2020":"Trough price during global pandemic",
#     "Q1 2022":"Russia invades Ukraine",
# }

# Save figure of price over time with gray regions indicating US recessions
figsize = (12, 6)
x = data.index
y = data['DCOILWTICO']

fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
ax.plot(x, y)
ax.set_xlim(data.index[0], data.index[-1])
ax.set_title("West Texas Intermediate (WTI) Daily Spot Price")
ax.set_xlabel("Date")
ax.set_ylabel("USD")
ax.fill_between(usrec.index, ax.get_ylim()[0], ax.get_ylim()[1],
                where=usrec["USREC"].values, color="k", alpha=0.1)
plt.grid(axis='y', color='lightgray')

fig.savefig("results/figures/wti_daily_spot_price.png")

""" 
This resamples daily data to be weekly. It takes the first value to maximize
the number of data points and to not average over intra-week variance.
Then it computes the excess returns percentage, which is used as a proxy for
volatility.
"""
resampled_data = data.resample('W').last()
percent_returns_data = resampled_data.pct_change().dropna() * 100 # Get weekly returns in percent change

"""
This plots the excess returns percentage with gray-shaded US recession indicators.
"""
fig, ax = plt.subplots(1, constrained_layout=True)

percent_returns_data.plot(title='Excess returns', figsize=(12, 3), ax=ax)
ax.fill_between(usrec.index, ax.get_ylim()[0], ax.get_ylim()[1], 
                where=usrec["USREC"].values, color="k", alpha=0.1)

fig.savefig("results/figures/wti_week_excess_returns_percent.png")

"""
This fits a Markov Switch Autoregressive model to 3 tasks, or regimes, that
differ in the variance of the data fit to. This is appropriate because we fit
the model to the proxy volatility, a measure of spread.
"""
k_regimes = 3
mod_kns = sm.tsa.MarkovRegression(percent_returns_data.dropna(), k_regimes=k_regimes, trend='n', switching_variance=True)
res_kns = mod_kns.fit(search_reps=200)

# Rename columns so that they are amenable for "i-th lowest variance regime"
new_col_titles = {0:1, 1:2, 2:3}
res_kns.smoothed_marginal_probabilities = res_kns.smoothed_marginal_probabilities.rename(
    columns=new_col_titles
)

"""This saves plots of the smoothed regime probabilities output by the MSA model 
fitted to the weekly excess return data. Note that for any given time step, the sum
of probabilities over all regimes equals 1. A plot of the excess returns is
included to visualize how regime probabilities change as excess return variance
changes.
"""
fig, axes = plt.subplots(k_regimes + 1, figsize=(12,8), constrained_layout=True)

for i, ax in enumerate(axes.flatten()[:-1]):
    axes[i].plot(res_kns.smoothed_marginal_probabilities[i+1])
    axes[i].set(title=f'Smoothed probability of {i+1}(th)-lowest variance regime for asset returns')
    axes[i].set_ylabel('Probability')
axes[k_regimes].plot(percent_returns_data)
axes[k_regimes].set(title='Excess returns')
ax.set_xlabel("Date")

fig.savefig("results/figures/wti_week_task_smooth_probs.png")

"""
This computes the task labels at each timestep. Each task label is the argmax
of probabilities over all regimes.
"""
smooth_probs = np.vstack([res_kns.smoothed_marginal_probabilities[i+1] for i in range(k_regimes)])
labels = np.argmax(smooth_probs, axis=0) + 1

"""
This saves a plot of the task labels at each timestep.
"""
fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
ax.plot(labels)
ax.set_title("Labels (for each point in time, the regime with argmax probability)")
ax.set_xlabel("Date")
ax.set_ylabel('Task')

fig.savefig("results/figures/wti_week_task_labels.png")

"""
This saves all four previous plots vertically stacked in the order they were
introduced as a single image.
"""
fig, axes = plt.subplots(4, figsize=(16,10), dpi=300, constrained_layout=True)

axes[0].plot(x, y)
axes[0].set_title("WTI Crude Oil Daily Spot Price")
axes[0].set_ylabel("USD/BBL")

axes[1].plot(percent_returns_data)
axes[1].set_ylabel('Percent change')
axes[1].set(title='Excess returns')

for i in range(k_regimes):
    axes[2].plot(res_kns.smoothed_marginal_probabilities[i+1], label=f'{i+1}')
    axes[2].set(title=f'Probability of different variance regimes for asset returns')
l = axes[2].legend(title='ith-lowest\nvariance regime', title_fontsize=10, fontsize=10, framealpha=0.5)
plt.setp(l.get_title(), multialignment='center')
axes[2].set_ylabel('Probability')

smooth_probs = np.vstack([res_kns.smoothed_marginal_probabilities[i+1] for i in range(k_regimes)])
labels = np.argmax(smooth_probs, axis=0) + 1
labels_df = pd.DataFrame(data=labels, index=percent_returns_data.index)
axes[3].plot(labels_df)
axes[3].set_ylabel('Regime')
axes[3].set_title("Labels (for each point in time, the regime with argmax probability)")
axes[3].yaxis.get_major_locator().set_params(integer=True)

for i in range(4):
    axes[i].set_xlim(data.index[0], data.index[-1])
axes[3].set_xlabel('Date')

for i in range(4):
    axes[i].fill_between(usrec.index, axes[i].get_ylim()[0], axes[i].get_ylim()[1], where=usrec["USREC"].values, color="k", alpha=0.1)

fig.savefig("results/figures/wti_all_plots.png")

"""
This outputs resampled data and corresponding task labels.
"""
data_copy['DATE_DT'] = pd.to_datetime(data_copy['DATE'])
data_copy['DCOILWTICO'] = pd.to_numeric(data_copy['DCOILWTICO'], errors='coerce')
data_copy = data_copy.set_index('DATE_DT')
data_copy = data_copy.drop('DATE', axis=1)

data_copy_resampled_weekly = data_copy.resample('W').last().dropna()

data_copy_resampled_weekly = data_copy_resampled_weekly.iloc[1:].squeeze().reset_index()

data_copy_resampled_weekly['TASK'] = labels

data_dir = "results/transformed_data"
output_filename = "wti_prices_tasks_resamp_week.csv"
output_filepath = os.path.join(data_dir, output_filename)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    
data.to_csv(output_filepath)