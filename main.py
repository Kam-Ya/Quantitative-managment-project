# Importing libraries, a bunch will be useless for now, just used these before
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# bad practice ik but for a solo dev not an issue? no one is is gonna be looking at this what am i saying
from helpers import *

df = pd.read_csv('Data/Sales_transactions_2022_2025.csv')

df = cleanData(df)

numeric = ['Customer_Age', 'Quantity', 'Unit_Price', 'Discount_Percentage', 'Inventory_Level', 'Delivery_Days', 'Order_Year']
categorical = ['Customer_Gender', 'Country', 'Customer_Segment', 'Sales_Channel', 'Product_Category', 'Payment_Method', 'Shipping_Method', 'Order_Status']




