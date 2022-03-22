import sklearn.datasets as skd
import numpy as np
from sklearn.impute import SimpleImputer as Simp
from sklearn.linear_model import LinearRegression as Lnr
import random
from pprint import pprint

cholesterol = skd.fetch_openml(name = 'cholesterol', version = 1, as_frame=True)
imp = Simp(missing_values=np.nan, strategy='mean')
imp.fit(cholesterol.data)
data = imp.transform(cholesterol.data)

indx = list(range(len(data)))
random.shuffle(indx)
train_indx, test_indx = indx[:201], indx[201:]
train_set, test_set = data[train_indx], data[test_indx]
train_target, test_target = cholesterol.target[train_indx], cholesterol.target[test_indx]


lin_reg = Lnr().fit(train_set, train_target)
train_score = round(lin_reg.score(train_set, train_target), 2)
target_score = round(lin_reg.score(test_set, test_target), 2)
print(train_score)
print(target_score)
