from pandas_datareader import data
import pandas as pd
import pandas_datareader.data as web
import datetime, os
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

today = datetime.date.today() - datetime.timedelta(days=1)
last_three_years = today - datetime.timedelta(days=3*365)

try:
    snp = web.DataReader("^GSPC", 'yahoo', last_three_years, today)
    print("Pulled SNP data")
except Exception as e:
    raise e

#Get returns (usual %age change)
returns = 100 * snp['Close'].pct_change().dropna()

#Also get differenced log returns, and squared log returns
diff_log_returns = np.log(snp['Close']/snp['Close'].shift(1)).dropna()
sq_diff_log_ret = np.square(diff_log_returns)

plt.plot(returns)
plt.show()

plt.plot(diff_log_returns)
plt.show()

plt.plot(sq_diff_log_ret)
plt.show()

#Check ACF - see meaningful correlation up to approx 17 lags
plot_acf(diff_log_returns)
plt.show()

#Check stationarity of log returns w/ ad fuller test
adfuller = adfuller(diff_log_returns)
print('ADF Statistic: %f' % adfuller[0])
print('p-value: %f' % adfuller[1])
print('Critical Values:')
for key, value in adfuller[4].items():
	print('\t%s: %.3f' % (key, value))

#p-val of 0 (lol) so reject H0: unit root, ie. log returns is stationary. Can therefore fit GARCH model.

# Split data into train/test for X-value
horizon = 7
train, test = diff_log_returns[:-horizon], diff_log_returns[-horizon:]
#Fit a GARCH(1,1) model to the data (adding in p,q=17 as motivated by ACF acc leads to higher AIC, so use more parsimonious model). Make the assumption that differenced log returns follow a normal dist.
garch_model_one = arch_model(train, mean='Constant', vol="Garch", p=1, o=0, q=1, dist="Normal")
output_one=garch_model_one.fit()
print(output_one.summary())

"""
We note here that the output has the following interpretation:
    omega: baseline variance
    alpha: MA term for yesterday on error^s ie. weighted white noise
    beta: effect of yesterdays vol on today's vol
    mu: expected return
"""

#Now, forecast variance for final week of dataset.

forecast = output_one.forecast(horizon=horizon, method="simulation", simulations=1000)
simulations = forecast.simulations

lines = plt.plot(simulations.values[-1,:,:].T, color='blue', alpha=0.01)
lines[0].set_label('Simulated paths')
plt.show()

print(np.percentile(simulations.values[-1,:,-1].T,5))
plt.hist(simulations.values[-1, :,-1],bins=50)
plt.title('Histogram of Returns')
plt.show()

"""
VaR is defined by:

    V = - sigma * phi^(-1)(alpha) * mu

where V=VaR, mu=mean of portfolio value, sigma = std dev of returns and phi inverse is the inverse of a cumulative standard normal dist. Therefore, use this formula to calc VaR using predicted volatility sigma.

"""
