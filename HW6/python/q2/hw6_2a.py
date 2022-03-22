import sklearn as sk
import numpy as np
import sklearn.datasets as skd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


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

plt.boxplot([lr_score, dtc_score], positions=[-1, 1],
            labels=["Logistic Regression", "Decision Tree Classifier"])
plt.ylim([.6, .7])
plt.ylabel("Cross Validation Scores with 10 Folds")
plt.title("Logistic Regression vs. Decision Tree Classifier\nCross Validation Scores")
plt.show()
