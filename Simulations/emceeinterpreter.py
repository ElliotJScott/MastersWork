import corner
import emcee
import numpy as np
import celerite
import autograd
from scipy.optimize import minimize


filenameStat = "mcdynmpaq.h5"
filenameDyn = "mcdynmpef.h5"
reader = emcee.backends.HDFBackend(filenameDyn)
#flatchain = reader.get_chain(flat=True)
discard = 10
thin = 1
samples = reader.get_chain(discard=discard, flat=True, thin=thin)
finsamp = reader.get_last_sample()
log_prob_samples = reader.get_log_prob(discard=discard, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=discard, flat=True, thin=thin)

#print("burn-in: {0}".format(burnin))
#print("thin: {0}".format(thin))
#print("flat chain shape: {0}".format(samples.shape))
#print("flat log prob shape: {0}".format(log_prob_samples.shape))
#print("flat log prior shape: {0}".format(log_prior_samples.shape))

#all_samples = np.concatenate(
#    (samples, log_prob_samples[:, None]), axis=1
#)

labels = ["r", "v", "M", "a"]
#labels += ["log prob"]

#corner.corner(samples, labels=labels)
figure = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
figure.savefig("yelafg")