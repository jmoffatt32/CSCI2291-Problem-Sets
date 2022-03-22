import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import show

#PART A:
D = np.loadtxt('python_files/datasets/data.csv', delimiter = ',', skiprows=1, usecols=tuple(range(2, 13)))
print(D.shape)

#PART B:
# matplotlib.pyplot.scatter(D[:,1], D[:,2])
# matplotlib.pyplot.xlabel("Mean Radius")
# matplotlib.pyplot.ylabel("Mean Texture")
# matplotlib.pyplot.xlim((0, max(D[:,1] + 5)))
# matplotlib.pyplot.ylim((0, max(D[:,1] + 5)))
# show()

#PART C:
meanRadBenign = D[D[:,0] == 0, 1]
meanRadMalignant = D[D[:,0] == 1, 1]
meanTextBenign = D[D[:,0] == 0, 2]
meanTextMalignant = D[D[:,0] == 1, 2]
benign = matplotlib.pyplot.scatter(meanRadBenign, meanTextBenign, c=['blue'])
malignant = matplotlib.pyplot.scatter(meanRadMalignant, meanTextMalignant, c=['red'])
matplotlib.pyplot.xlabel("Mean Radius")
matplotlib.pyplot.ylabel("Mean Texture")
matplotlib.pyplot.xlim((min(D[:,1]) - 5, max(D[:,1] + 5)))
matplotlib.pyplot.ylim((min(D[:,1]) - 5, max(D[:,1] + 5)))
matplotlib.pyplot.legend([benign, malignant], ["Benign", "Malignant"])
show()
