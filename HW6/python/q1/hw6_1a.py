import sklearn as sk
import numpy as np
import sklearn.datasets as skd
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from pprint import pprint

np.set_printoptions(precision=3, suppress=False)

higgs_dataset = skd.fetch_openml("higgs", version=1, as_frame=True)
data = higgs_dataset.data
target = higgs_dataset.target

imp = Imputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

scaler = StandardScaler()
data = scaler.fit_transform(data)

lr = skl.LogisticRegression().fit(data, target)
score = lr.score(data, target)
pprint(score)
