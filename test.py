import numpy as np

A = np.array([
    [10, 20],
    [30, 40]
])

B = np.array([
    [7],
    [8]
])

# print(f"A*B = \n{A.dot(B)}")
# print(f"AT*B = \n{A.T.dot(B)}")
# print(f"det(A) = {np.linalg.det(A)}")

arr1 = np.ones((10, 128))
arr2 = np.ones((10, 1))

arr3 = np.sum(arr2 * arr1, axis=0)
print(arr3.shape)

arr3 = np.reshape(arr3, (128, 1))
print(arr3.shape)