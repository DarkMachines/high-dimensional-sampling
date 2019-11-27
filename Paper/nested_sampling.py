from anesthetic import NestedSamples
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import numpy
import tqdm

def loglikelihood(x):
    """Example non-trivial loglikelihood

    - Constrained zero-centered correlated parameters x0 and x1,
    - half-constrained x2 (exponential).
    - unconstrained x3 between 0 and 1
    - x4 is a slanted top-hat distribution between 2 and 4

    """
    x0, x1 = x[:]
    sigma0, sigma1 = 0.1, 0.1 
    mu0, mu1 = 0.7, 0.7
    eps = 0.9
    x0 = (x0-mu0)/sigma0
    x1 = (x1-mu1)/sigma1

    logl_A = -numpy.log(2*numpy.pi*sigma0*sigma1*(1-eps**2)**0.5) - (x0**2 - 2*eps*x0*x1 + x1**2)/(1-eps**2)/2

    x0, x1 = x[:]
    sigma0, sigma1 = 0.1, 0.1 
    mu0, mu1 = 0.3, 0.3
    eps = -0.9
    x0 = (x0-mu0)/sigma0
    x1 = (x1-mu1)/sigma1

    logl_B = -numpy.log(2*numpy.pi*sigma0*sigma1*(1-eps**2)**0.5) - (x0**2 - 2*eps*x0*x1 + x1**2)/(1-eps**2)/2

    return logsumexp([logl_B,logl_A]) - numpy.log(2)


def ns_sim(ndims=2, nlive=100, ndead=700):
    """Brute force Nested Sampling run"""
    numpy.random.seed(0)
    low=(0, 0)
    high=(1,1)
    live_points = numpy.random.uniform(low=low, high=high, size=(nlive, ndims))
    live_likes = numpy.array([loglikelihood(x) for x in live_points])
    live_birth_likes = numpy.ones(nlive) * -numpy.inf

    dead_points = []
    dead_likes = []
    birth_likes = []
    for _ in tqdm.tqdm(range(ndead)):
        i = numpy.argmin(live_likes)
        Lmin = live_likes[i]
        dead_points.append(live_points[i].copy())
        dead_likes.append(live_likes[i])
        birth_likes.append(live_birth_likes[i])
        live_birth_likes[i] = Lmin
        while live_likes[i] <= Lmin:
            live_points[i, :] = numpy.random.uniform(low=low, high=high, size=ndims) 
            live_likes[i] = loglikelihood(live_points[i])
    return dead_points, dead_likes, birth_likes, live_points, live_likes, live_birth_likes

data, logL, logL_birth, live, live_logL, live_logL_birth = ns_sim()

ns = NestedSamples(data=data, logL=logL, logL_birth=logL_birth)

fig, axes = plt.subplots(1,4, sharex=True, sharey=True, figsize=(5.95,5.95/3.8))

for i, ax in enumerate(axes):
    j = i*100
    points = ns.iloc[:j]
    ax.plot(points[0], points[1], '.', ms=4, label='dead points')
    points = ns.live_points(ns.logL[j])
    ax.plot(points[0], points[1], '.', ms=4, label='live points')
    ax.plot(0.7, 0.7, 'k+')
    ax.plot(0.3, 0.3, 'k+')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])

fig.tight_layout()
fig.savefig('nested_sampling.pdf')
