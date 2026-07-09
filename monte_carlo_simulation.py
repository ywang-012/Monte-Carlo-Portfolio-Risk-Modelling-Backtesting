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
test_prices=test_data["Close"]

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

#plot
fig, axs = plt.subplots(2, 2, figsize=(16,10))
for i in range(len(tickers)):
    axs[0, 0].plot(price_paths[:, :, i].T, alpha=0.1)
    axs[0, 0].set_title(f"Monte Carlo Simulated Price Paths for all assets (1 Year)")
    axs[0, 0].set_xlabel("Trading Days")
    axs[0, 0].set_ylabel("Price")
    axs[0, 0].grid(True)

    axs[0, 1].hist(final_prices[:, i], bins=50, alpha=0.4, label=f"{tickers[i]}", edgecolor='black')
    axs[0, 1].set_title(f"Distribution of Final Simulated Prices for all assets (1 Year)")
    axs[0, 1].set_xlabel("Final Price")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].grid(True)

    axs[1, 0].hist(final_returns[:, i], bins=50, alpha=0.4, label=f"{tickers[i]}", edgecolor='black')
    axs[1, 0].set_title(f"Distribution of Final Simulated Returns for all assets(1 Year)")
    axs[1, 0].set_xlabel("Final Return")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(True)

    axs[1, 1].bar(tickers[i], prob_losses[i])
    axs[1, 1].set_title(f"Probability of Loss for all assets(1 Year)")
    axs[1, 1].set_ylabel("Probability")
    axs[1, 1].grid(True)



    risk_text = (
        f"Initial price: ${price_paths[0, 0, i]:.2f}\n"
        f"Mean Final Price: ${mean_final_prices[i]:.2f}\n"
        f"Value at Risk (95%): ${Var_95[i]:.2f}\n"
        f"Expected Shortfall (95%): ${ES_95[i]:.2f}"
        f"Prob of loss: {prob_losses[i]:.2%}"
        f"annualized mean return: {mu_values[i]:.2%}\n"
        f"annualized volatility: {sigma_values[i]:.2%}\n"

    )

axs[1, 1].text(
    0.95, 0.95, risk_text,
    transform=axs[1, 1].transAxes,
    fontsize=11,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8)

)

axs[0,1].legend()
axs[1,0].legend()

plt.tight_layout()

plt.savefig(
    "simulation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()