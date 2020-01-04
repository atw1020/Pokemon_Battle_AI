
import numpy as np
import sys

class leak:

    def __init__(self):

        self.init_arrays()
        self.fill_arrays()

    def init_arrays(self):

        self.data = np.empty((50000, 100000))  # all the memory should be pre-allocated here

        print(sys.getsizeof(self.data) / (50000 * 100000))
        print(sys.getsizeof(self.data) / (1024 ** 3))

    def fill_arrays(self):

        for i in range(self.data.shape[0]):

            # activity monitor says I've allocated like 4 GB around 10% completion

            if i % (self.data.shape[0] // 100) == 0:
                print(i / (self.data.shape[0] // 100), "% complete", sep="")

            self.data[i] = np.random.rand(self.data.shape[1])


test = leak()
