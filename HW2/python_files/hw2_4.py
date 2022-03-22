import numpy as np


D = np.loadtxt('python_files/datasets/data.csv', delimiter = ',', skiprows=1, usecols=tuple(range(2, 13)))
desc = D[:,[1,2,3,4,5,6,7,8,9,10]]
meanDesc = np.mean(desc, axis=0)

for size in range(0, 6):
    N = 10 * (2 ** size) #sample size
    sample = np.empty((0,10))
    normSamp = []
    for _ in range(10001):
        sample = desc[np.random.choice(desc.shape[0], size=N, replace=True), :]
        meanSamp = np.mean(sample, axis=0)
        dist = abs(meanSamp - meanDesc)
        normSamp.append(np.linalg.norm(dist))
    normSamp = np.array(normSamp)
    print(f"({N}, {np.mean(normSamp)})")
    


    

