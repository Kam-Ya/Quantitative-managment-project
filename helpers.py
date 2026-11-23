import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def cleanData(df):

    print("Cleaning data:")
    print(f"Original Length: {len(df)}")

    # this should clean all rows with missing data except ones where missing data is expected or reasonable.
    df_clean = df.dropna(subset=[col for col in df.columns if col not in ['Promotion_Code', 'Customer_Rating', 'Return_Reason']])
    print(f"Drop NaN: {str(len(df_clean))}")

    # remove duplicate transactions for regressiion
    df_clean = df_clean.drop_duplicates(subset='Transaction_ID')

    print(f"Remove duplicates: {str(len(df_clean))}")

    # remove outliers in age
    clean = df_clean[
        df_clean['Customer_Age'].between(18,100)
        & df_clean['Quantity'].gt(0)
        & df_clean["Discount_Percentage"].le(100)
        ].copy()
    print(f"Remove outliers: {len(clean)}")

    return clean

def formula(cat, data):
    # making a string formula for statsmodel imports
    rhs = []
    for i in cat:
        rhs.append(f"C({i})")
    for i in data:
        rhs.append(f"{i}")

    return("Profit ~ " + " + ".join(rhs))

def elimination(data, cat, numeric, alpha=0.05):
    step = 0
    while True:
        model = smf.ols(formula(cat, numeric), data=data).fit()
        pvals = sm.stats.anova_lm(model, typ=2)["PR(>F)"].drop("Residual")
        worst = pvals.idxmax()


        left = len(cat) + len(numeric)

        if pvals[worst] <= alpha or left == 1:
            return model, cat + numeric

        name = worst.replace("C(", "").strip(")")
        step += 1

        print(f"Step {step}: removing {name} (p = {pvals[worst]:.4f}), "
              f"adj R² = {model.rsquared_adj:.4f}")
        if name in cat:
            cat.remove(name)
        else:
            numeric.remove(name)

def model_table(model):
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    y = pd.Series(model.model.endog)

    table = pd.DataFrame({
        'B': model.params,
        'Std Error': model.bse,
        't': model.tvalues,
        'p': model.pvalues,
    })

    # Standardized beta for every column (numeric and dummy)
    table['Std Beta'] = table['B'] * X.std(ddof=1) / y.std(ddof=1)
    table.loc['Intercept', 'Std Beta'] = np.nan

    # VIF for every non-intercept column
    table['VIF'] = pd.Series({
        col: variance_inflation_factor(X.values, i)
        for i, col in enumerate(X.columns) if col != 'Intercept'
    })
    return table
