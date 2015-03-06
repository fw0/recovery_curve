import itertools
import numpy as np
import scipy.stats
import itertools
import pdb

f_base = object
obj_base = object

class dist(obj_base):

    def loglik(self, x):
        pass

    def sample(self):
        pass

def np_beta(ms, phis):
    ss = apply(lambda phi: 1.0/phi - 1.0, phis)
    alphas = itertools.imap(lambda s,m: 1+s*m, ms, ss)
    betas = itertools.imap(lambda s,m: 1+s*(1.0-m), ms, ss)
    return np.random.beta(a=alphas, b=betas)

class beta_mode_phi_dist(dist):

    def __init__(self, mode, phi):
        self.mode, self.phi = mode, phi
        self.a, self.b = beta_mode_phi_to_a_b(mode, phi)
        self.horse = scipy.stats.beta(a=self.a, b=self.b)

    def sample(self):
        return self.horse.rvs()

    def loglik(self, x):
        import pdb
        return self.horse.logpdf(x)

def beta_mode_phi_to_a_b(mode, phi):
    s = (1.0 / phi) - 1.0
    a = 1 + s * mode
    b = 1 + s*(1.0 - mode)
    return a,b
        
def sample_beta_phi_mode(phi, mode):
    a, b = beta_mode_phi_to_a_b(mode, phi)
    return scipy.stats.beta.rvs(a=a, b=b)
        
class beta_mean_beta_dist(dist):

    def __init__(self, mean, beta):
        self.mean, self.beta = mean, beta
        alpha = beta * mean / (1.0 - mean)
        self.horse = scipy.stats.beta(a=alpha, b=beta)

    def sample(self):
        return self.horse.rvs()

    def loglik(self, x):
        return self.horse.logpdf(x)

class gamma_mode_phi_dist(dist):

    def __init__(self, mode, phi):
        a, scale = gamma_mode_phi_to_a_scale(mode, phi)
        self.horse = scipy.stats.gamma(a=a, scale=scale)

    def sample(self):
        return self.horse.rvs()

    def loglik(self, x):
        return self.horse.logpdf(x)

class constant_dist(dist):

    def __init__(self, val):
        self.val = val

    def sample(self, *args):
        return self.val
        
    def batch_sample(self, *args):
        return self.val
        
    def loglik(self, *args):
        return 0

class frozen_dist(dist):

    def __init__(self, horse):
        self.horse = horse

    def sample(self, *args):
        return self.horse(*args)
    
def sample_gamma_phi_mode(phi, mode):
    a, scale = gamma_mode_phi_to_a_scale(mode, phi)
    return scipy.stats.gamma.rvs(a=a, scale=scale)
        
def gamma_mode_phi_to_a_scale(mode, phi):
    alpha = 1.0 / phi
    beta = (alpha - 1.0) / mode
    a = alpha
    scale = 1. / beta
    return a, scale

def sample_truncated_exponential(l):
    while 1:
        ans = scipy.stats.expon.rvs(scale = 1. / l)
        if ans < 1.:
            return ans
    
def f(s, a, b, c, t):
    try:
        return s * ((1.0 - a) - b * (1.0 - a) * np.exp(-t / c))
    except:
        pdb.set_trace()

def scaled_f(a, b, c, t):
    return f(a, b, c, 1.0, t)

def logistic(x):
    return 1. / (1.+np.exp(-x))

def logit(x):
    return np.log(x) - np.log(1-x)

def unflatten(l, lengths):
    """
    returns list of arrays
    """
    cumsum = np.cumsum(lengths)
    lows = itertools.chain([0], cumsum[:-1])
    highs = cumsum
    return [l[low:high] for (low, high) in itertools.izip(lows, highs)]
