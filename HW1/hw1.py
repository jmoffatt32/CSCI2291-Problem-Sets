import numpy as np
import matplotlib as plt

#---QUESTION 1---#
print("\nQUESTION 1:\n")
#Part A:

# print("Part A:\n")
# L = [4, -4, 6, 8, -2, 7]
# print(min([x ** 2 for x in L]))

# #Part B:

# print("\nPart B:\n")

# D = np.array([4, -4, 6, 8, -2, 7])
# print(D[-3])

# #Part C:

# print("\nPart C:\n")
# print(   sum([n**3 for n in range(-50, (10**4 + 1))])   )

#Part D:

# print("\nPart D:\n")
# print(   [n for n in range(100) if n**4 > 500 * n]   )


# #Part E:

print("\nPart E:\n")

dic = { 1 : 'red', 
        2 : 'black', 
        3 : 'red', 
        4 : 'black', 
        5 : 'red', 
        6 : 'black', 
        7: 'red'}
print(  len([ dic[key] for key in dic if dic[key] != 'red'])  )
print('\n')

#--QUESTION 3--#
print("-------------------------")
print("\nQuestion 3:\n")

#Part A:

# print("Part A: \n")


# def generate_largest_a():
#     m = 1
#     while 10 ** m < (1000 * (m ** 6)):
#         m += 1
#     m -= 1
#     print(m)
# generate_largest_a()

# #Part B:
# print("\nPart A: \n")

# def generate_largest_b():
#     m = 100
#     while 1000 - (100 * m) + (5 * m ** 2) - ((1 / 15) * (m ** 3)) < 500:
#         m -= 1
#     print(m)
# generate_largest_b()

