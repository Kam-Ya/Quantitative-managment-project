# Importing libraries, a bunch will be useless for now, just used these before
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import statsmodels.api as sm

# bad practice ik but for a solo dev not an issue? no one is is gonna be looking at this what am i saying
from helpers import *

df = pd.read_csv('Data/Sales_transactions_2022_2025.csv')

df = cleanData(df)
df['Price_x_Qty'] = df['Unit_Price'] * df['Quantity']

numeric = ['Customer_Age', 'Quantity', 'Unit_Price', 'Discount_Percentage', 'Inventory_Level', 'Delivery_Days', 'Order_Year', 'Price_x_Qty']
categorical = ['Customer_Gender', 'Country', 'Customer_Segment', 'Sales_Channel', 'Product_Category', 'Payment_Method', 'Shipping_Method', 'Order_Status']

for c in categorical:
    df[c] = df[c].str.strip().str.lower()

modelFull = smf.ols(formula(categorical, numeric), data=df).fit()
modelFinal, kept = elimination(df, categorical, numeric)

print("Kept: ", kept)
print(modelFinal.summary())

pd.set_option('display.float_format', '{:,.4f}'.format)
print(model_table(modelFinal))


comparison = pd.DataFrame({
    'Full model': [modelFull.df_model, modelFull.rsquared, modelFull.rsquared_adj, modelFull.aic, modelFull.bic],
    'Final model': [modelFinal.df_model, modelFinal.rsquared, modelFinal.rsquared_adj, modelFinal.aic, modelFinal.bic],
}, index=['Predictors (df)', 'R²', 'Adjusted R²', 'AIC', 'BIC'])
print(comparison)

# Do the removed variables add anything jointly? (nested F-test)
print(sm.stats.anova_lm(modelFinal, modelFull))

resid = modelFinal.resid
fitted = modelFinal.fittedvalues
exog = modelFinal.model.exog

# ---------- Plots ----------
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# 1. Linearity: residuals vs fitted (with a smoothed trend line)
ax[0, 0].scatter(fitted, resid, alpha=0.25, s=8)
smooth = sm.nonparametric.lowess(resid, fitted, frac=0.3)
ax[0, 0].plot(smooth[:, 0], smooth[:, 1], color='red')
ax[0, 0].axhline(0, color='black', lw=0.8)
ax[0, 0].set(title='Residuals vs Fitted (linearity)', xlabel='Fitted', ylabel='Residual')

# 2. Homoscedasticity: scale-location
ax[0, 1].scatter(fitted, np.sqrt(np.abs(resid / resid.std())), alpha=0.25, s=8)
ax[0, 1].set(title='Scale-Location (homoscedasticity)', xlabel='Fitted',
             ylabel='√|standardized residual|')

# 3. Normality: Q-Q plot
sm.qqplot(resid, line='45', fit=True, ax=ax[1, 0], markersize=3)
ax[1, 0].set_title('Normal Q-Q (normality)')

# 4. Normality: histogram
ax[1, 1].hist(resid, bins=80)
ax[1, 1].set(title='Residual histogram', xlabel='Residual')

plt.tight_layout()
plt.savefig('diagnostics.png', dpi=150)
plt.close()

# ---------- Formal tests ----------
# Linearity: Ramsey RESET (H0: model is correctly specified / linear)
reset = linear_reset(modelFinal, power=2, test_type='fitted', use_f=True)
print(f"RESET:           F = {float(reset.statistic):.2f}, p = {float(reset.pvalue):.4g}")

# Homoscedasticity: Breusch-Pagan (H0: constant variance)
lm, lm_p, f, f_p = het_breuschpagan(resid, exog)
print(f"Breusch-Pagan:   LM = {lm:.2f}, p = {lm_p:.4g}")

# Independence: Durbin-Watson (about 2 = no autocorrelation)
print(f"Durbin-Watson:   {durbin_watson(resid):.3f}")

# Normality: Jarque-Bera (H0: normal), plus skew and kurtosis
jb, jb_p, skew, kurt = jarque_bera(resid)
print(f"Jarque-Bera:     JB = {jb:.0f}, p = {jb_p:.4g}, skew = {skew:.2f}, kurtosis = {kurt:.1f}")

# ---------- Independence: shared Order_ID ----------
order_ids = df.loc[resid.index, 'Order_ID']
sizes = order_ids.value_counts()
print(f"\nOrders with more than one line: {(sizes > 1).sum()} of {len(sizes)} "
      f"({(sizes > 1).mean():.1%})")

# Do residuals cluster within orders? (one-way ANOVA on multi-line orders)
multi = order_ids[order_ids.map(sizes) > 1]
if multi.nunique() > 1:
    groups = [resid[idx].values for _, idx in multi.groupby(multi).groups.items()]
    F, p = stats.f_oneway(*groups)
    print(f"Residuals by Order_ID: F = {F:.2f}, p = {p:.4g}")
