import sklearn.datasets as skd
from sklearn.impute import SimpleImputer as Simp
from sklearn.linear_model import LinearRegression as Lnr
import numpy as np
from numpy.linalg import pinv 
from pprint import pprint

np.set_printoptions(suppress=True)

cholesterol = skd.fetch_openml(name = 'cholesterol', version = 1, as_frame=True)
imp = Simp(missing_values=np.nan, strategy='mean')
imp.fit(cholesterol.data)
data = imp.transform(cholesterol.data) #X

data_mod = np.append(data, np.ones((303,1)), axis = 1)      #X
target = cholesterol.target                                 #y

left_term = pinv(np.dot(data_mod.T, data_mod))
right_term = (np.dot(data_mod.T, target))
c = np.dot(left_term, right_term)
print("Calculated c-column:")
print(c)

lin_reg = Lnr(fit_intercept=True).fit(data, target)
c_column = lin_reg.coef_
print("sklearn Linear Regression c-column:")
print(c_column)

diff = np.around((c[:13] - c_column), decimals = 10)
pprint(diff)