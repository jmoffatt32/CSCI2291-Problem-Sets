import sklearn.datasets as skd
from sklearn.impute import SimpleImputer as Simp
from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.linear_model import LinearRegression as Lnr
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
import random

cholesterol = skd.fetch_openml(name = 'cholesterol', version = 1, as_frame=True)
imp = Simp(missing_values=np.nan, strategy='mean')
imp.fit(cholesterol.data)
data = imp.transform(cholesterol.data)
target = cholesterol.target

poly = Poly(degree = 2)
quad_data = poly.fit_transform(data)
print(f"{'Original Data Shape:' :<45}{data.shape}")
print(f"{'Polynomial Data Shape:' :<45}{quad_data.shape}")

lin_reg = Lnr().fit(quad_data, cholesterol.target)
score = lin_reg.score(quad_data, cholesterol.target)
print(f"{'Linear Regression on Polynomial Data Score:' :<45}{score}")


train_scores = []
test_scores = []
for i in range(2500):
    indx = list(range(len(quad_data)))
    random.shuffle(indx)
    indx_train, indx_test = indx[:202], indx[202:]
    quad_train, quad_test = quad_data[indx_train], quad_data[indx_test]
    target_train, target_test = target[indx_train], target[indx_test]

    lin_reg_2 = Lnr().fit(quad_train, target_train)
    train_scores.append(lin_reg_2.score(quad_train, target_train))
    test_scores.append(lin_reg_2.score(quad_test, target_test))
print(f"{'Median R^2 Score on Training Set:' :<45}{round(np.median(train_scores), 3)}")
print(f"{'Median R^2 Score on Testing Set:' :<45}{round(np.median(test_scores), 3)}")









