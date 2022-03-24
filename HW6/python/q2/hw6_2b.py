import sklearn as sk
import numpy as np
import sklearn.datasets as skd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind


np.set_printoptions(precision=3, suppress=False)

higgs_dataset = skd.fetch_openml("higgs", version=1, as_frame=True)
data = higgs_dataset.data
target = higgs_dataset.target

imp = Imputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

scaler = StandardScaler()
data = scaler.fit_transform(data)

lr = LogisticRegression().fit(data, target)
dtc = DecisionTreeClassifier().fit(data, target)
lr_score = cross_val_score(lr, data, target, cv=10)
dtc_score = cross_val_score(dtc, data, target, cv=10)

lr_score_stand = scaler.fit_transform(lr_score.reshape(-1, 1))[:, 0]
dtc_score_stand = scaler.fit_transform(dtc_score.reshape(-1, 1))[:, 0]

test_stand = ttest_ind(lr_score_stand, dtc_score, equal_var=True)
print(f"{'t-statistic:' :<20}{test_stand[0]}")
print(f"{'p-value:' :<20}{test_stand[1]}")

# high p-value, very difficult to distinguish if one or the other certainly has a higher corelation
# essentialy suggesting that the result of the lr_score being higher is due to random chance (the randomly selected data values)
