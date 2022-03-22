#IMPORTS
import sklearn as sk
import sklearn.datasets as skd
from sklearn.impute import SimpleImputer as Simp
from sklearn.preprocessing import PolynomialFeatures as Poly
from sklearn.linear_model import LinearRegression as Lnr
import numpy as np
from numpy.linalg import pinv 
from pprint import pprint
import matplotlib.pyplot as plt
import math, random

np.set_printoptions(suppress=True)

#QUESTION 1:
print('--------------------------------------------------------------------------------\nQUESTION 1:\n')
diabetes = skd.load_diabetes()
data = diabetes['data']
cov_matrix = np.cov(data, rowvar = False)

#Part A:
print("\nPart A:\n--------------------")
print(f"{'Covariance Matrix Shape:' :<45}{cov_matrix.shape}")

#Part B:
print("\nPart B:\n--------------------")
bmi_idx, hdl_idx = 0, 0
for i, feat in enumerate(diabetes.feature_names):
    if feat == 'bmi':
        bmi_idx = i
    if feat == 's3':
        hdl_idx = i
print(f"{'Correlation of bmi to hdl:' :<45}{cov_matrix[bmi_idx][hdl_idx] :.5f}")

#Part C:
print("\nPart C:\n--------------------")
bmi_mean = np.mean(data[:,2])
print(f"{'bmi mean:' :<45}{bmi_mean :.5f}")
hdl_above_bmi_mean = data[:,6][data[:,2] > bmi_mean]
hdl_below_bmi_mean = data[:,6][data[:,2] < bmi_mean]
median_above = np.median(hdl_above_bmi_mean)
median_below = np.median(hdl_below_bmi_mean)
print(f"{'Median of hdl for bmi values above mean:' :<45}{median_above :.5f}")
print(f"{'Median of hdl for bmi values below mean:' :<45}{median_below :.5f}")

#PART D:
print("\nPart D:\n--------------------")
s = math.sqrt((np.var(hdl_above_bmi_mean) + np.var(hdl_below_bmi_mean)))
d = round(abs(((np.mean(hdl_above_bmi_mean)) - (np.mean(hdl_below_bmi_mean))) / s), 3)
print(f"{'Cohens d Value:' :<45}{d}")

#QUESTION 2
print('\n--------------------------------------------------------------------------------\nQUESTION 2:\n')

#PART A:
print("\nPart A:\n--------------------")
cholesterol = skd.fetch_openml(name = 'cholesterol', version = 1, as_frame=True)
imp = Simp(missing_values=np.nan, strategy='mean')
imp.fit(cholesterol.data)
data = imp.transform(cholesterol.data)

lin_reg = Lnr().fit(data, cholesterol.target)
score = round(lin_reg.score(data, cholesterol.target), 2)
c_column = lin_reg.predict(data)
print(f"{'R^2 Goodness of Fit Score:' :<45}{score}")

predict_target = lin_reg.predict(data).astype(int)
real_target = cholesterol.target.astype(int)
num_correct = np.count_nonzero(predict_target == real_target)
frac_correct = round(num_correct / (predict_target.shape[0]), 2)
print(f"{'Fraction of Correct Predictions:' :<45}{frac_correct}")

#Part B:
print("\nPart B:\n--------------------")
train_scores, test_scores = [], []
for i in range(5000):
    indx = list(range(len(data)))
    random.shuffle(indx)
    train_indx, test_indx = indx[:201], indx[201:]
    train_set, test_set = data[train_indx], data[test_indx]
    train_target, test_target = cholesterol.target[train_indx], cholesterol.target[test_indx]
    lin_reg = Lnr().fit(train_set, train_target)
    train_scores.append(lin_reg.score(train_set, train_target))
    test_scores.append(lin_reg.score(test_set, test_target))

print(f"{'Median R^2 Score on Training Set:' :<45}{round(np.median(train_scores), 2)}")
print(f"{'Median R^2 Score on Testing Set:' :<45}{round(np.median(test_scores), 2)}")

#QUESTION 3:
print('\n--------------------------------------------------------------------------------\nQUESTION 3:\n')

#PART A:
print("\nPart A:\n--------------------")
data_mod = np.append(data, np.ones((303,1)), axis = 1)      #X
target = cholesterol.target                                 #y

left_term = pinv(np.dot(data_mod.T, data_mod))
right_term = (np.dot(data_mod.T, target))
c = np.dot(left_term, right_term)
print("Calculated c-column:")
print(np.around(c, decimals = 2))

lin_reg = Lnr(fit_intercept=True).fit(data, target)
c_column = lin_reg.coef_
print("sklearn.LinearRegression c-column:")
print(np.around(c_column, decimals = 2))

diff = np.around((c[:13] - c_column), decimals = 20)
print("Difference Vector Between sklearn.LinearRegression and Custom Regression:")
print(diff)
print(f"{'Sum of Difference Vector Terms:' :<45}{sum(diff)}")

#QUESTION 4:
print('\n--------------------------------------------------------------------------------\nQUESTION 4:\n')

#PART A:
print("\nPart A:\n--------------------")
poly = Poly(degree = 2)
quad_data = poly.fit_transform(data)
print(f"{'Original Data Shape:' :<45}{data.shape}")
print(f"{'Polynomial Data Shape:' :<45}{quad_data.shape}")

#Part B:
print("\nPart B:\n--------------------")
lin_reg = Lnr().fit(quad_data, cholesterol.target)
score = lin_reg.score(quad_data, cholesterol.target)
print(f"{'Linear Regression on Polynomial Data Score:' :<45}{score}")

train_scores = []
test_scores = []
for i in range(5000):
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
