import matplotlib
import numpy as np
from matplotlib.pyplot import show


x = np.random.normal(3, 1, 1000)

#PART A:
mean = round(np.mean(x), 3)
std = round(np.std(x), 3)
print(f"{'Mean:'}{mean :>6}{'STD:' :>15}{std :>6}")

#PART B:
xStand = (x - np.mean(x)) / (np.std(x))
meanStand = round(np.mean(xStand), 3)
stdStand = round(np.std(xStand), 3)
print(f"{'Mean:'}{meanStand :>6}{'STD:' :>15}{stdStand :>6}")

#PART C:
xSquares = x ** 2
matplotlib.pyplot.boxplot(xSquares)
matplotlib.pyplot.title("Graph of Squares")
show()

#PART D:
median = round(np.percentile(xSquares, 50), 3)
firstQ = round(np.percentile(xSquares, 25), 3)
thirdQ = round(np.percentile(xSquares, 75), 3)
print(f"{'First Quartile' :<20}{'Median' :<20}{'Third Quartile' :<20}")
print(f"{firstQ :<20}{median :<20}{thirdQ :<20}")

