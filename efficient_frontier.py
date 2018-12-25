import pandas as pd
import numpy as np
import pandas_datareader as pdr
import pointcloud as pc

N = 10
def get_tickers(N = 10):

	companylistNASDAQ = pd.read_csv("companylistNASDAQ.csv")
	tickers =  companylistNASDAQ.sort_values(by = ['MarketCap'])[::-1][:N]["Symbol"].tolist()
	return tickers

	

def get_closing_prices(tickers):
	#L = pdr.get_data_yahoo(tickers)["Close"]
	#L = L.fillna(0)
	#L.to_csv("data.csv")
	L = pd.read_csv("data.csv")[tickers]
	L = L.fillna(0)
	return L

def returns(stock):
    return (stock - stock.shift(1))[1:]

def stock_returns_metric(tickers, L):
	#L = get_closing_prices(tickers)
	ticker_pairs = [(x,y) for x in tickers for y in tickers if tickers.index(x) < tickers.index(y)]
	metric = {}
	for pair in ticker_pairs:
		X = returns(L[pair[0]]).values
		Y = returns(L[pair[1]]).values
		metric[pair] = np.sqrt(np.mean(np.square(np.absolute(X-Y)))) #l2 
        	#metric[pair] = np.mean(np.absolute(X-Y))  #l_inf
	return metric

def EV(tickers,L):
	#L = get_closing_prices(tickers)
	out = []
	for ticker in tickers:
        	out.append([np.var(returns(L[ticker])), np.mean(returns(L[ticker]))])
	return np.array(out)


def portfolios(iter, n):
	portfolios = []
	for i in range(iter):
		r = np.random.random(n)
		s = (r/sum(r).tolist())
		portfolios.append(s)
	return portfolios

def portfolios_pairs(iter, n):
	portfolios = []
	pairs = [[i,j] for i in range(n) for j in range(n) if i<j]
	for i in range(iter):
		for pair in pairs:
			portfolio = [0]*n
			r = np.random.random(2)
			s = (r/sum(r).tolist())
			portfolio[pair[0]] = s[0]
			portfolio[pair[1]] = s[1]
			portfolios.append(portfolio)
	return portfolios


def covariance_matrix(tickers,L):
	#L = get_closing_prices(tickers)
	return np.cov(returns(L[tickers]), rowvar = False)


def portfolio_returns(tickers,L,portfolio):
	#L = get_closing_prices(tickers)
	alpha = np.mean(returns(L[tickers])).values.tolist()
	Sigma = covariance_matrix(tickers,L)
	E = np.dot(np.array(alpha), np.matrix(portfolio).transpose())
	V = np.dot(np.dot(np.array(portfolio), Sigma), np.matrix(portfolio).transpose())
	Sharpe = E[0,0]/np.sqrt(V[0,0])
	return E[0,0],V[0,0], Sharpe

def portfolios_returns(tickers,L, portfolios):
	#L = get_closing_prices(tickers)
	EV = []
	for portfolio in portfolios:
		EV.append(portfolio_returns(tickers, L, portfolio))
	return np.array(EV)

def std_basis(n): 
    A = np.identity(n)
    return A.tolist()
    
def  EF(tickers,L):
	EV = portfolios_returns(tickers, L, portfolios(iter = 1000, n = len(tickers)))
	EV_basis = portfolios_returns(tickers, L, std_basis(n = len(tickers)))
	#EV_pairs = portfolios_returns(tickers, L, portfolios_pairs(iter = 1000, n = len(tickers)))

	from matplotlib import pyplot as plt
	from matplotlib import cm
	fig = plt.figure(figsize = (15,7))
	plt.scatter(EV[:,1], EV[:,0], marker = "o", c = EV[:,2], s = 50, alpha = 1, edgecolors = 'black', linewidths = 1, cmap = cm.coolwarm)
	#plt.scatter(EV_pairs[:,1], EV_pairs[:,0], marker = "o", c = EV_pairs[:,2], s = 50, alpha = 1, edgecolors = 'black', cmap = cm.coolwarm)

	plt.scatter(EV_basis[:,1], EV_basis[:,0], marker = "^", s = 100, c = EV_basis[:,2])
	for i,txt in enumerate(tickers):
		plt.annotate(txt, (EV_basis[:,1][i], EV_basis[:,0][i]))
	
	plt.xlabel("Returns Variance/Volatility")
	plt.ylabel("Returns Mean")
	plt.title("Efficient Frontier")
	plt.colorbar()
	plt.show()

def EF_pairwise(tickers,L):

	EV = portfolios_returns(tickers, L, portfolios_pairs(iter = 1000, n = len(tickers)))
	EV_basis = portfolios_returns(tickers, L, std_basis(n = len(tickers)))

	from matplotlib import pyplot as plt
	from matplotlib import cm
	fig = plt.figure(figsize = (15,7))
	plt.scatter(EV[:,1], EV[:,0], marker = "o", c = 'black', s = 5, alpha = 1, edgecolors = 'black', linewidths = 1, cmap = cm.coolwarm)
	plt.scatter(EV_basis[:,1], EV_basis[:,0], marker = "^", s = 100, c = 'black')
	for i,txt in enumerate(tickers):
		plt.annotate(txt, (EV_basis[:,1][i], EV_basis[:,0][i]))
	
	plt.xlabel("Returns Variance/Volatility")
	plt.ylabel("Returns Mean")
	plt.title("Efficient Frontier")
	#plt.colorbar()
	plt.show()
	





