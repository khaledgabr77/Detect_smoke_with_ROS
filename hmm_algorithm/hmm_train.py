#!/usr/bin/python

from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

 
def plotTimeSeries(Q, hidden_states, ylabel, filename):
 
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    xs = np.arange(len(Q))+1
    masks = hidden_states == 0
    ax.scatter(xs[masks], Q[masks], c='g', label='Normal State')
    masks = hidden_states == 1
    ax.scatter(xs[masks], Q[masks], c='r', label='Smoky State')
    ax.plot(xs, Q, c='k')
     
    ax.set_xlabel('Room')
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    fig.clf()
 
    return None
 
def fitHMM(Q, nSamples):
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q,[len(Q),1]))
     
    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(Q,[len(Q),1]))
 
    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
 
    # find log-likelihood of Gaussian HMM
    Prob = model.score(np.reshape(Q,[len(Q),1]))
 
    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)
 
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states
 
    return hidden_states, mus, sigmas, P, Prob, samples
 
# load annual flow data for the Colorado River near the Colorado/Utah state line
Clusters = np.loadtxt('/home/khaled/Downloads/HMM_ReleaseV1.0/khaled_log.txt')
 
# log transform the data and fit the HMM
#logQ = np.log(AnnualQ)
hidden_states, mus, sigmas, P, Prob, samples = fitHMM(Clusters, 100)
plt.switch_backend('agg') # turn off display when running with Cygwin
plotTimeSeries(Clusters, hidden_states, 'Number of Clusters', 'StateTseries_Log2.png')

