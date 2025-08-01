import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix

def optimize_portfolio(mean_returns, cov_matrix):
    n = len(mean_returns)
    weights = cp.Variable(n)
    portfolio_return = mean_returns.values @ weights
    portfolio_risk = cp.quad_form(weights, cov_matrix.values)
    objective = cp.Maximize(portfolio_return / cp.sqrt(portfolio_risk))
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return weights.value

def portfolio_metrics(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return / port_volatility
    return port_return, port_volatility, sharpe_ratio

def monte_carlo_var(portfolio_returns, confidence_level=0.05, simulations=10000):
    simulated = np.random.choice(portfolio_returns, size=(simulations,), replace=True)
    return np.percentile(simulated, 100 * confidence_level)

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    data = fetch_data(tickers, start='2023-01-01', end='2024-01-01')
    returns, mean_returns, cov_matrix = calculate_returns(data)
    opt_weights = optimize_portfolio(mean_returns, cov_matrix)

    port_ret, port_vol, sharpe = portfolio_metrics(opt_weights, mean_returns, cov_matrix)
    var_95 = monte_carlo_var(returns.mean(axis=1))

    print("\n--- Portfolio Optimization Results ---")
    for t, w in zip(tickers, opt_weights):
        print(f"{t}: {w:.2%}")
    print(f"Expected Annual Return: {port_ret:.2%}")
    print(f"Volatility: {port_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"95% VaR (1-day): {var_95:.2%}")
