import pandas as pd
import matplotlib.pyplot as plt

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

