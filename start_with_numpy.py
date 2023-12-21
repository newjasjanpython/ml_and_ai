import numpy as np


print("Arrays and dtype numpy")

simple_np_array = np.array([13, 5, 6, 48, 4, 6])
print("Simple array", simple_np_array)


simple_np_range = np.arange(10)
print("Simple range", simple_np_range)


simple_random_np = np.random.random((3, 3))
print("Simple random", simple_random_np)


simple_random_numbers = np.random.random((3, 3))
simple_random_numbers[:] *= 10
simple_random_numbers = simple_random_numbers.astype(int) # change to number type int64
print("Simple random numbers", simple_random_numbers)


simple_2d_array = np.ndarray((5, 5), np.int32)
print("Simple 2d array", simple_2d_array)


simple_image_type = np.zeros((500, 500, 3), np.int8) ## maybe ones or empty
# shape 500x500 image with 3 RGB channels
print("Simple image", simple_image_type)

