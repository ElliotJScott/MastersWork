import corner
import numpy as np

ndim, nsamples = 4, 1000
np.random.seed(42)
samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
figure = corner.corner(samples)
figure.savefig("testfig")