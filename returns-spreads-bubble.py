import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from statsmodels.api import OLS

W = 5 # window for averaging earnings
L = 10 # lag for earnings
print('Window for averaged earnings = ', W)
data = pd.read_excel('century.xlsx', sheet_name = 'price')
vol = data['Volatility'].values[1:]
div = data['Dividends'].values[1:] #annual dividends
index = data['Price'].values #annual index values
N = len(vol)
data0 = pd.read_excel('century.xlsx', sheet_name = 'earnings')
cpi = data0['CPI'].values #annual consumer price index
earn = data0['Earnings'].values #annual earnings
TR = [np.log(index[k+1] + div[k]) - np.log(index[k]) for k in range(N)]

spreads = pd.DataFrame({})
spreads['BAA-AAA'] = data['BAA'] - data['AAA']
spreads['AAA-Long'] = data['AAA'] - data['Long']
spreads['Long-Short'] = data['Long'] - data['Short']
rdiv = cpi[-1]*div/cpi[L:]
rearn = cpi[-1]*earn/cpi
rindex = cpi[-1]*index/cpi[L-1:]
rTR = [np.log(rindex[k+1] + rdiv[k]) - np.log(rindex[k]) for k in range(N)]

wealth = np.append(np.array([1]), np.exp(np.cumsum(TR))) # nominal wealth
rwealth = np.append(np.array([1]), np.exp(np.cumsum(rTR))) # real wealth
cumrearn = [sum(rearn[k:k+W])/W for k in range(N+1)] # cumulative earnings 
rgrowth = np.diff(np.log(cumrearn))
IDY = rTR - rgrowth
cumIDY = np.append(np.array([0]), np.cumsum(IDY))
DF = pd.DataFrame({'const' : 1/vol, 'trend' : -np.array(range(N))/vol, 'Bubble' : cumIDY[:-1]/vol})
Regression = OLS(IDY/vol, DF).fit()
coefficients = Regression.params
intercept = coefficients['const']
trend_coeff = coefficients['trend']
bubble_coeff = coefficients['Bubble']
c = trend_coeff / bubble_coeff
print('Long-term difference between returns and growth = ', c)
Valuation = cumIDY - c * np.array(range(N+1))

DFreg = pd.DataFrame({'const' : 1/vol, 'Measure' : Valuation[:-1]/vol, 'Vol' : 1})
for key in spreads:
    DFreg[key] = spreads[key].iloc[:-1]/vol

print('Total Returns')
Reg = OLS(rTR/vol, DFreg).fit()
print('\n Full Regression Real Returns \n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Real Returns Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("total-real-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Real Returns Residuals \n ACF for Original Values')
plt.savefig("total-real-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Real Returns Residuals \n ACF for Absolute Values')
plt.savefig("total-real-aacf.png", dpi=300, bbox_inches='tight')
plt.close()

Reg = OLS(TR/vol, DFreg).fit()
print('\n Full Regression Nominal Returns \n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Nominal Returns Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("total-nominal-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Nominal Returns Residuals \n ACF for Original Values')
plt.savefig("total-nominal-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Nominal Returns Residuals \n ACF for Absolute Values')
plt.savefig("total-nominal-aacf.png", dpi=300, bbox_inches='tight')
plt.close()

print('Regressions for Price Returns')
Reg = OLS(np.diff(np.log(rindex))/vol, DFreg).fit()
print('\n Full Regression Real Returns \n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Real Returns Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("price-real-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Real Returns Residuals \n ACF for Original Values')
plt.savefig("price-real-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Real Returns Residuals \n ACF for Absolute Values')
plt.savefig("price-real-aacf.png", dpi=300, bbox_inches='tight')
plt.close()

Reg = OLS(np.diff(np.log(index))/vol, DFreg).fit()
print('\n Full Regression Nominal Returns \n')
print(Reg.summary())
residuals = Reg.resid
stderr = np.std(residuals)
print('Analysis of residuals')
print('Standard deviation = ', stderr)
print('Shapiro-Wilk and Jarque-Bera p = ', stats.shapiro(residuals)[1], stats.jarque_bera(residuals)[1])
aresiduals = abs(residuals)
qqplot(residuals, line = 's')
plt.title('Nominal Returns Residuals \n Quantile Quantile Plot vs Normal')
plt.savefig("price-nominal-qqplot.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(residuals)
plt.title('Nominal Returns Residuals \n ACF for Original Values')
plt.savefig("price-nominal-acf.png", dpi=300, bbox_inches='tight')
plt.close()
plot_acf(aresiduals)
plt.title('Nominal Returns Residuals \n ACF for Absolute Values')
plt.savefig("price-nominal-aacf.png", dpi=300, bbox_inches='tight')
plt.close()