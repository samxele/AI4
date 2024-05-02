import numpy as np

print("Hello World!")

# Basic ways to make np arrays
# Default type is np.float64
print(np.array([1, 2, 3]))
print(np.arange(2, 9, 2))
print(np.linspace(0, 10, num = 5))

# 2D arrays - fancy!
print(np.array([[1, 2], [3, 4], [5, 6]]))

# RNG time
rng = np.random.default_rng()
print(rng.integers(5, size=(2, 4)))