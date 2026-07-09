from cProfile import label

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
np.random.seed(42) #for reproducibility
tickers = ["NVDA", "AAPL", "GOOGL", "AMZN"]
weights = np.array([0.4, 0.3, 0.2, 0.1])  # Example weights for the portfolio
data = yf.download(["NVDA", "AAPL", "GOOGL", "AMZN"], start="2020-01-01", end="2025-01-01") #data frame
log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
test_start = "2025-01-01"
test_end = "2026-01-01"
test_data = yf.download(tickers, start=test_start, end=test_end)
test_prices=test_data["Close"] #latest closing prices for each ticker #data frame

mu = log_returns.mean() * 252  # Annualized mean return, series (4,)
sigma = log_returns.std() * np.sqrt(252)  # Annualized volatility, series (4,)
mu_values = mu.to_numpy() #ndarray (4,)
sigma_values = sigma.to_numpy() #ndarray (4,)
S0 = data["Close"].iloc[-1] # latest closing prices for each ticker

T = 1
N = 252
dt = T / N
M = 10000

price_paths = np.zeros((M, N + 1, len(tickers))) #ndarray (10000, 253, 4)
price_paths[:, 0, :] = data["Close"].iloc[-1].values #Initialize all paths at the latest closing prices

correlation_matrix = log_returns.corr()
L = np.linalg.cholesky(correlation_matrix)
Z_independent = np.random.normal(0, 1, size=(M, N, len(tickers)))
Z_correlated  = Z_independent @ L.T



for t in range(1, N+1):
    price_paths[:, t, :] = price_paths[:, t - 1, :] * np.exp(
        (mu_values - 0.5 * sigma_values**2) * dt + sigma_values * np.sqrt(dt) * Z_correlated[:, t - 1, :]

    )
final_prices = price_paths[:, -1, :] #ndarray (10000, 4)
final_returns = (final_prices - price_paths[:, 0, :]) / price_paths[:, 0, :]
initial_prices = price_paths[:, 0, :]
losses = initial_prices - final_prices #ndarray (10000, 4)
prob_losses = np.mean(final_prices - initial_prices < 0, axis=0)
mean_final_prices = np.mean(final_prices, axis=0)


Var_95 = np.percentile(losses, 95, axis=0)
ES_95 = np.array([
    losses[:, i][losses[:, i] >= Var_95[i]].mean()
    for i in range(len(tickers))
])

#portfolio simulation
portfolio_initial_price = np.sum(initial_prices * weights, axis=1)
portfolio_final_prices = np.sum(final_prices * weights, axis=1)
portfolio_losses = portfolio_initial_price - portfolio_final_prices
portfolio_prob_loss = np.mean(portfolio_final_prices - portfolio_initial_price < 0)
portfolio_mean_final_price = np.mean(portfolio_final_prices)
portfolio_return = (portfolio_final_prices - portfolio_initial_price) / portfolio_initial_price
lower_95 = np.percentile(portfolio_return, 2.5)
upper_95 = np.percentile(portfolio_return, 97.5)
mean_return = np.mean(portfolio_return)



#portfolio backtesting
actual_initial_prices = test_data["Close"].iloc[-1].to_numpy()
actual_final_prices = test_prices.iloc[-1].to_numpy()

actual_asset_returns = (actual_final_prices - actual_initial_prices) / actual_initial_prices

actual_portfolio_return = actual_asset_returns @ weights

portfolio_backtest_results = pd.DataFrame({
    "Actual Final Price": actual_final_prices,
    "Mean Simulated Final Price": mean_final_prices,
    "Lower 95% Simulated Price": lower_95,
    "Upper 95% Simulated Price": upper_95,
    "Actual Inside 95% Interval": (
        (actual_final_prices >= lower_95) &
        (actual_final_prices <= upper_95)
    ),
    "Prediction Error (%)": (
        (actual_final_prices - mean_final_prices) / mean_final_prices
    )
}, index=tickers)




#plot
plt.figure(figsize=(10,6))
plt.hist(portfolio_return, bins=50, alpha=0.7, label='simulated portfolio returns', edgecolor='black')
plt.axvline(actual_portfolio_return, color = 'red', linestyle='-', linewidth = 2.5, label=f'actual portfolio ({actual_portfolio_return:.2%})')

plt.axvline(lower_95,linestyle=":",linewidth=2, label=f'Lower 95% Bound ({lower_95:.2%})')

plt.axvline(upper_95,linestyle=":",linewidth=2,label=f'Upper 95% Bound ({upper_95:.2%})')
plt.axvline(mean_return, linestyle="--", linewidth=2, label=f'Mean simulated ({mean_return:.2%})')


plt.title("Backtest: Simulated vs Actual Portfolio Return in 2025")
plt.xlabel("Portfolio Return")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

