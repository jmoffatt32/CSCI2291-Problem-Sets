from sklearn import datasets
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

digits = datasets.load_digits(as_frame = True)
projection = MDS(n_components = 2, )
projected_digits = projection.fit_transform(digits.data) #returns np.array

color_map = {
    '0' : 'red', '1' : 'blue',
    '2' : 'yellow', '3' : 'orange',
    '4' : 'maroon', '5' : 'green',
    '6' : 'saddlebrown', '7' : 'cyan',
    '8' : 'slategray', '9' : 'lime'
}

for target in digits.target_names:
    sorted_data = projected_digits[digits.target == target] 
    plt.scatter(sorted_data[:,0], sorted_data[:,1], c= [color_map[str(target)]] )
plt.show()



