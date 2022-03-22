from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn import datasets



breast_cancer = datasets.load_breast_cancer(as_frame = True)
benign = breast_cancer.data.loc[breast_cancer.target == 0]
malignant = breast_cancer.data.loc[breast_cancer.target == 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

projection = MDS(n_components = 3)
projected_benign = projection.fit_transform(benign)
projected_malignant = projection.fit_transform(malignant)

benign_scatter = ax.scatter(projected_benign[:,0], projected_benign[:,1], projected_benign[:,2], c=['blue'] )
malignant_scatter = ax.scatter(projected_malignant[:,0], projected_malignant[:,1], projected_malignant[:,2], c=['red'] )
plt.show()