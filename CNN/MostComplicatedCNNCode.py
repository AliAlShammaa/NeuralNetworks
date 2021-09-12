## Back-to-back Convolutonal layer backprop algorithm in long form
import numpy as np

b , a = np.array()

for i in range(3):
    for p in range(5):
        for q in range(5):
            b[p, q, i, p:p + 3 , q:q + 3,:,:] = a