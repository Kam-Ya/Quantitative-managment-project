# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
classification_report, confusion_matrix,
roc_auc_score, roc_curve
)

from sklearn.preprocessing import LabelEncoder

# bad practice ik but for a solo dev not an issue? no one is is gonna be looking at this what am i saying
from helpers import *


dataset = # TODO upload dataset
columnNames = # TODO make a list

data = pd.read_csv() 


