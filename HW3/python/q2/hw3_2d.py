from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn import datasets

digits = datasets.load_digits(as_frame = True)
projection = MDS(n_components = 3, )
projected_digits = projection.fit_transform(digits.data) #returns np.array

color_map = {
    '0' : 'red', '1' : 'blue',
    '2' : 'yellow', '3' : 'orange',
    '4' : 'maroon', '5' : 'green',
    '6' : 'saddlebrown', '7' : 'cyan',
    '8' : 'slategray', '9' : 'lime'
}

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for target in digits.target_names:
    sorted_data = projected_digits[digits.target == target] 
    ax.scatter(sorted_data[:,0], sorted_data[:,1], sorted_data[:,2], c = [color_map[str(target)]] )
plt.show()