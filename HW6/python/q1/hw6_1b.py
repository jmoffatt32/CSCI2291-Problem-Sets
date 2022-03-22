import sklearn as sk
import numpy as np
import sklearn.datasets as skd
import sklearn.linear_model as skl
import sklearn.preprocessing as skp
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer as Imputer
from pprint import pprint

np.set_printoptions(precision=3, suppress=False)

higgs_dataset = skd.fetch_openml("higgs", version=1, as_frame=True)
data = higgs_dataset.data
target = higgs_dataset.target

imp = Imputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

scaler = skp.StandardScaler()
data = scaler.fit_transform(data)

lr = skl.LogisticRegression().fit(data, target)
cross_score = cross_val_score(lr, data, target, cv=10)
pprint(cross_score)
print(f"{'Mean: ' :<20}{np.mean(cross_score)}")
print(f"{'STD Deviation: ' :<20}{np.std(cross_score)}")
