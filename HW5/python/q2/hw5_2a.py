import sklearn.datasets as skd
import numpy as np
from sklearn.impute import SimpleImputer as Simp
from sklearn.linear_model import LinearRegression as Lnr
from pprint import pprint

cholesterol = skd.fetch_openml(name = 'cholesterol', version = 1, as_frame=True)
imp = Simp(missing_values=np.nan, strategy='mean')
imp.fit(cholesterol.data)
data = imp.transform(cholesterol.data)

lin_reg = Lnr().fit(data, cholesterol.target)
score = round(lin_reg.score(data, cholesterol.target), 2)
c_column = lin_reg.predict(data)
print(f"{'R^2 Goodness of Fit Score:' :<45}{score}")

predict_target = lin_reg.predict(data).astype(int)
real_target = cholesterol.target .astype(int)
num_correct = np.count_nonzero(predict_target == real_target)
frac_correct = round(num_correct / (predict_target.shape[0]), 2)
print(f"{'Fraction of Correct Predictions:' :<45}{frac_correct}")




