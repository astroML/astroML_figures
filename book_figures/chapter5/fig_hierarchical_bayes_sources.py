"""
An illustration of Hierarchical Bayes modeling for a Gaussian burst with background
------------------------------------------
Figure 5.28

An example of hierarchical Bayes modeling: estimate the position, amplitude,
width and background for a Gaussian burst. Then assume that a large population
of such bursts almost fully constrains the background and compare the marginal
distributions of amplitude for the two cases. Code modeled after Figure 10.25.
"""
# Author: Zeljko Ivezic
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2019)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib import pyplot as plt

# Hack to fix import issue in older versions of pymc
import scipy
import scipy.misc
scipy.derivative = scipy.misc.derivative
import pymc

from astroML.plotting.mcmc import plot_mcmc
from astroML.decorators import pickle_results

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

def gauss(t, b0, A, ssig, T):
    """Gaussian Point Spread Function"""
    y = np.empty(t.shape)
    y.fill(b0)
    y += A * np.exp(-(t - T)**2/2/ssig**2)
    return y

#----------------------------------------------------------------------
# Set up toy dataset
np.random.seed(0)

N = 20
b0_true = 10
A_true = 10
ssig_true = 15.0
alpha_true = 1/ssig_true**2/2
T_true = 50
sigma = 3.0
t = np.linspace(2,98,N)
y_true = gauss(t, b0_true, A_true, ssig_true, T_true)
y_obs = np.random.normal(y_true, sigma)

#----------------------------------------------------------------------
# Set up MCMC sampling
A = pymc.Uniform('A', 0, 50, value=50 * np.random.random())
T = pymc.Uniform('T', 0, 100, value=100 * np.random.random())
log_ssig = pymc.Uniform('log_ssig', -10, 10, 0.0)
fit_b = True
if (fit_b):
    b0 = pymc.Uniform('b0', 0, 50, value=50 * np.random.random())
else:
    # very strong prior for b0
    b0 = pymc.Uniform('b0', 0.999*b0_true, 1.001*b0_true, value=b0_true)

# uniform prior on log(ssig)
@pymc.deterministic
def ssig(log_ssig=log_ssig):
    return np.exp(log_ssig)

@pymc.deterministic
def y_model(t=t, b0=b0, A=A, ssig=ssig, T=T):
    return gauss(t, b0, A, ssig, T)

y = pymc.Normal('y', mu=y_model, tau=sigma ** -2, observed=True, value=y_obs)
model = dict(b0=b0, A=A, T=T, log_ssig=log_ssig,
             ssig=ssig, y_model=y_model, y=y)

#----------------------------------------------------------------------
# Run the MCMC sampling
@pickle_results('gauss.pkl')
def compute_MCMC_results(niter=25000, burn=4000):
    S = pymc.MCMC(model)
    S.sample(iter=niter, burn=burn)
    traces = [S.trace(s)[:] for s in ['b0', 'A', 'T', 'ssig']]

    M = pymc.MAP(model)
    M.fit()
    fit_vals = (M.b0.value, M.A.value, M.ssig.value, M.T.value)

    return traces, fit_vals

traces, fit_vals = compute_MCMC_results()
#------------------------------------------------------------
# Plot the results
fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(bottom=0.1, top=0.95,
                    left=0.1, right=0.95,
                    hspace=0.05, wspace=0.05)

true = [b0_true, A_true, T_true, ssig_true]
labels = ['$b_0$', '$A$', '$T$', r'$\sigma$']
limits = [(0.0, 19.0), (0, 19), (35, 65), (0.0, 60.0)]

# This function plots multiple panels with the traces
plot_mcmc(traces, labels=labels, limits=limits, true_values=true, fig=fig,
          bins=30, colors='k')

# plot the model fit
ax = fig.add_axes([0.5, 0.75, 0.45, 0.20])
t_fit = np.linspace(0, 100, 101)
y_fit = gauss(t_fit, *fit_vals)

ax.scatter(t, y_obs, s=9, lw=0, c='k')
ax.plot(t_fit, y_fit, '-k')
ax.plot(t, y_true, ls="--", c="r", lw=1)
ax.set_xlim(0, 100)
ax.set_xlabel('$time$')
ax.set_ylim(0, 25)
ax.set_ylabel(r'$Counts_{\rm obs}$')

# plot the marginal amplitude distributions with b0 fit and assumed known
ax = fig.add_axes([0.77, 0.45, 0.18, 0.20])
ax.plot(t, y_true, ls="--", c="r", lw=1)
ax.set_xlim(6, 20)
ax.set_xlabel('$A$')
ax.set_ylim(0, 0.35)
ax.set_ylabel(r'$p(A)}$')

tb = traces[0]
tA = traces[1]
# simulate HB constraint from "other" sources as
tAb = tA[np.abs(tb-b0_true)<0.5]
# compare marginal amplitude distributions
ax.hist(tAb, 15, normed=True, histtype='stepfilled', alpha=0.8)
ax.hist(tA, 15, normed=True, histtype='stepfilled', alpha=0.5)
ax.plot([10.0, 10.0], [0.0, 1.0], ls="--", c="r", lw=1)

plt.savefig('HBsources.png')
plt.show()
