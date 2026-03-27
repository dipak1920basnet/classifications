from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np 

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
