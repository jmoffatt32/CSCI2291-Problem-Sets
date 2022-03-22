import numpy as np
import timeit

#--QUESTION 1--#

#PART A:

def square_of_array(a):
    return a ** 0.5

def square_of_array_2(a):
    for i, term in enumerate(np.nditer(a)):
        a[(i)] = term ** 0.5
    return a

#PART B:

a = np.random.rand(10 ** 4)

start1 = timeit.default_timer()
for i in range(0,1001):
    a1 = square_of_array(a)
end1 = timeit.default_timer()
start2 = timeit.default_timer()
for i in range(0,1001):
    a2 = square_of_array_2(a)
end2 = timeit.default_timer()

time1 = end1 - start1
time2 = end2 - start2
time_ratio = time1 / time2

print(f"{'Vectorized Form Time:' :<30}{round(time1, 3) :<6.3f}")
print(f"{'Iterative Form Time:' :<30}{round(time2, 3) :<6.3f}")
print(f"{'Vectorized/Iterative Ratio:' :<30}{round(time_ratio, 3) :<6.3f}")

#PART B:


