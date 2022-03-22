import sklearn as sk
import sklearn.datasets as skd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import math

diabetes = skd.load_diabetes()
data = diabetes.data
cov_matrix = np.cov(data, rowvar = False)
pprint(diabetes)

#Part A:
pprint(cov_matrix.shape)

#Part B:
#bmi = 2
#hdl = s3 = 6
print(f"{'Correlation of bmi to hdl:' :<45}{cov_matrix[2][6] :.5f}")
bmi_mean = np.mean(data[:,2])
print(f"{'bmi mean:' :<45}{bmi_mean :.5f}")
hdl_above_bmi_mean = data[:,6][data[:,2] > bmi_mean]
hdl_below_bmi_mean = data[:,6][data[:,2] < bmi_mean]
median_above = np.median(hdl_above_bmi_mean)
median_below = np.median(hdl_below_bmi_mean)
print(f"{'Median of hdl for bmi values above mean:' :<45}{median_above :.5f}")
print(f"{'Median of hdl for bmi values below mean:' :<45}{median_below :.5f}")

s = math.sqrt((np.var(hdl_above_bmi_mean) + np.var(hdl_below_bmi_mean)))

d = abs(((np.mean(hdl_above_bmi_mean)) - (np.mean(hdl_below_bmi_mean))) / s)

print(f"{'Cohens d Value:' :<45}{d}")

