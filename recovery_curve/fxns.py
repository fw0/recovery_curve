import recovery_curve.recovery_curve.utils as recovery_utils
import numpy as np
import scipy.stats
import functools
from collections import namedtuple
import itertools
import os
import pystan
import pdb
import python_utils.python_utils.basic as basic_utils
import pandas as pd

import recovery_curve

STAN_FOLDER = '%s/%s' % (os.path.dirname(recovery_curve.__file__), 'stan_files')

B_phis = namedtuple('B_phis', ['B_a', 'B_b', 'B_c', 'phi_a', 'phi_b', 'phi_c'])

class B_phis_dist(recovery_utils.dist):

    def __init__(self, s_a, s_b, s_c, phi_a_dist, phi_b_dist, phi_c_dist, K):
        self.s_a, self.s_b, self.s_c, self.phi_a_dist, self.phi_b_dist, self.phi_c_dist = s_a, s_b, s_c, phi_a_dist, phi_b_dist, phi_c_dist
        self.K = K

    def sample(self):
        return B_phis(scipy.stats.norm.rvs(self.s_a,size=self.K), scipy.stats.norm.rvs(self.s_b,size=self.K), scipy.stats.norm.rvs(self.s_c,size=self.K), self.phi_a_dist.sample(), self.phi_b_dist.sample(), self.phi_c_dist.sample())

class B_phis_dist_fitter(object):

    def __init__(self, s_a, s_b, s_c, phi_a_dist, phi_b_dist, phi_c_dist):
        self.s_a, self.s_b, self.s_c, self.phi_a_dist, self.phi_b_dist, self.phi_c_dist = s_a, s_b, s_c, phi_a_dist, phi_b_dist, phi_c_dist
    
    def fit(self, recovery_X_train, ys_ns_train):
        K = len(iter(recovery_X_train.x_ns).next())
        return B_phis_dist(self.s_a, self.s_b, self.s_c, self.phi_a_dist, self.phi_b_dist, self.phi_c_dist, K)

    
abc = namedtuple('abc', ['a','b','c'])

class a_dist(recovery_utils.dist):

    def __init__(self, pop_val):
        self.pop_val = pop_val
    
    def sample(self, x, B_a, phi_a):
        mode = recovery_utils.logistic(recovery_utils.logit(self.pop_val) + np.dot(B_a, x))
        return recovery_utils.sample_beta_phi_mode(phi_a, mode)

b_dist = a_dist

class c_dist(recovery_utils.dist):

    def __init__(self, pop_val):
        self.pop_val = pop_val

    def sample(self, x, B_c, phi_c):
        mode = np.exp(np.log(self.pop_val) + np.dot(B_c, x))
        return recovery_utils.sample_gamma_phi_mode(phi_c, mode)

abc = namedtuple('abc', ['a','b','c'])

def abc_dist_sample_helper(a_dist, b_dist, c_dist, B_phis, x):
    return abc(a_dist.sample(x, B_phis.B_a, B_phis.phi_a), b_dist.sample(x, B_phis.B_b, B_phis.phi_b), c_dist.sample(x, B_phis.B_c, B_phis.phi_c))
                    
class abc_ns_dist(recovery_utils.dist):

    def __init__(self, a_dist, b_dist, c_dist):
        self.a_dist, self.b_dist, self.c_dist = a_dist, b_dist, c_dist
    
    def sample(self, x_ns, B_phis):
        return [abc_dist_sample_helper(self.a_dist, self.b_dist, self.c_dist, B_phis, x) for x in x_ns]
        return map(lambda x: abc(self.a_dist.sample(x, B_phis.B_a, B_phis.phi_a, self.b_dist.sample(x, B_phis.B_b, B_phis.phi_b), self.c_dist.sample(x, B_phis.B_c, B_phis.phi_c)), x_ns))

class noise_dist(recovery_utils.dist):

    def __init__(self, theta, p, phi_m):
        self.theta, self.p, self.phi_m = theta, p, phi_m
    
    def sample(self, mode):
        return recovery_utils.sample_beta_phi_mode(self.phi_m, mode) if np.random.random() > self.theta else (1 if np.random.random() < self.p else 0)

class noise_dist_dist(recovery_utils.dist):

    def __init__(self, l_m):
        self.l_m = l_m

    def sample(self):
        return noise_dist(scipy.stats.uniform.rvs(), scipy.stats.uniform.rvs(), recovery_utils.sample_truncated_exponential(self.l_m))

class beta_noise_dist(recovery_utils.dist):

    def __init__(self, phi_m):
        self.phi_m = phi_m
    
    def sample(self, mode):
        return recovery_utils.sample_beta_phi_mode(self.phi_m, mode)

class beta_noise_dist_dist(recovery_utils.dist):

    def __init__(self, l_m):
        self.l_m = l_m

    def sample(self):
        return beta_noise_dist(recovery_utils.sample_truncated_exponential(self.l_m))

def simulate_xs(N, K, stdev):
    return np.array([scipy.stats.norm.rvs(loc=0, scale=stdev, size=K) for n in xrange(N)])
    
class ys_dist(recovery_utils.dist):

    def sample(self, noise_dist, s, abc, ts):
#        print noise_dist,s,abc,ts
        return np.array(map(noise_dist.sample, map(functools.partial(recovery_utils.f, s, *abc), ts)))
    
class ys_ns_dist(recovery_utils.dist):

    def __init__(self):
        self.horse = ys_dist()
    
    def sample(self, noise_dist, s_ns, abc_ns, ts_ns):
        """
        return list of arrays
        """
        return [self.horse.sample(noise_dist, s, abc, ts) for (s, abc, ts) in itertools.izip(s_ns, abc_ns, ts_ns)]

everything = namedtuple('everything',['B_phis','noise_dist','abc_ns','ys_ns'])

everything_with_test = namedtuple('everything',['B_phis','noise_dist','abc_ns','ys_ns','abc_ns_test','ys_ns_test'])
    
class everything_dist(recovery_utils.dist):

    def __init__(self, B_phis_dist, abc_ns_dist, noise_dist_dist):
        self.B_phis_dist, self.abc_ns_dist, self.noise_dist_dist = B_phis_dist, abc_ns_dist, noise_dist_dist
    
    def sample(self, s_ns, x_ns, ts_ns):
        B_phis = self.B_phis_dist.sample()
        noise_dist = self.noise_dist_dist.sample()
        abc_ns = self.abc_ns_dist.sample(x_ns, B_phis)
        ys_ns = ys_ns_dist().sample(noise_dist, s_ns, abc_ns, ts_ns)
        return everything(B_phis, noise_dist, abc_ns, ys_ns)

class everything_with_test_dist(everything_dist):

    def __init__(self, everything_dist):
        self.everything_dist = everything_dist
    
    def sample(self, s_ns, x_ns, ts_ns, s_ns_test, x_ns_test, ts_ns_test):
        B_phis, noise_dist, abc_ns, ys_ns = self.everything_dist.sample(s_ns, x_ns, ts_ns)
        abc_ns_test = self.everything_dist.abc_ns_dist.sample(x_ns_test, B_phis)
        ys_ns_test = ys_ns_dist.sample(noise_dist, s_ns_test, abc_ns_test, ts_ns_test)
        return everything_with_test(B_phis, noise_dist, abc_ns, ys_ns, abc_ns_test, ys_ns_test)

def everything_dist_to_param_dict(everything_dist):
    pop_a = everything_dist.abc_ns_dist.a_dist.pop_val
    pop_b = everything_dist.abc_ns_dist.b_dist.pop_val
    pop_c = everything_dist.abc_ns_dist.c_dist.pop_val

    s_a = everything_dist.B_phis_dist.s_a
    s_b = everything_dist.B_phis_dist.s_b
    s_c = everything_dist.B_phis_dist.s_c
    l_a = everything_dist.B_phis_dist.phi_a_dist.horse.keywords['l']
    l_b = everything_dist.B_phis_dist.phi_b_dist.horse.keywords['l']
    l_c = everything_dist.B_phis_dist.phi_c_dist.horse.keywords['l']
    l_m = everything_dist.noise_dist_dist.l_m

    return {\
            'pop_a':pop_a,\
            'pop_b':pop_b,\
            'pop_c':pop_c,\
            's_a':s_a,\
            's_b':s_b,\
            's_c':s_c,\
            'l_a':l_a,\
            'l_b':l_b,\
            'l_c':l_c,\
            'l_m':l_m,\
            }

traces_base = namedtuple('traces',['permuted','unpermuted','data'])


class traces(traces_base):

    def param_to_chain_trace(self, param, chain):

        # figure out ending point of params
        param_lens = np.zeros(len(self.permuted.keys()))
        for (i,key) in enumerate(self.permuted.keys()):
            try:
                param_lens[i] = self.permuted[key].shape[1]
            except IndexError:
                param_lens[i] = 1
        ends = np.cumsum(param_lens)

        # figure out index of param
        idx = None
        for i in range(len(self.permuted.keys())):
            if self.permuted.keys()[i] == param:
                idx = i
                break
        assert idx is not None
        if idx == 0:
            start_pos = 0
        else:
            start_pos = ends[idx-1]
        end_pos = ends[idx]

        return self.unpermuted[:,chain,start_pos:end_pos]

    @property
    def num_chains(self):
        return self.unpermuted.shape[1]
        
    def param_to_trace(self, param):
        return np.concatenate([self.param_to_chain_trace(param,chain) for chain in xrange(self.num_chains)])

    def param_dim(self, param):
        return self.permuted[param].shape[1]
    
    def gelman_statistic(self, param):
        within = np.var(self.param_to_trace(param))
        between = np.mean(np.array([np.var(self.param_to_chain_trace(param, i)) for i in xrange(self.num_chains)]))
        return (within, between)

    def trace_figs(self, params, thin=1):
        figs = []
        for param in params:
            dim = self.param_dim(param)
            n_rows, n_cols = (1,1) if dim == 1 else (3,1)
            param_figs, param_axes = basic_utils.get_grid_fig_axes(n_rows, n_cols, dim)
            assert len(param_axes) == dim
            for (i,param_ax) in enumerate(param_axes):
                param_ax.set_title('%s %d' % (param,i))
                for j in xrange(self.num_chains):
                    component_trace = self.param_to_chain_trace(param,i)[:,j]
                    param_ax.plot(component_trace[0:len(component_trace):thin], alpha=0.5)
        return figs
            
def get_everything_dist_traces_helper(n_steps, random_seed, num_chains, everything_dist, s_ns, x_ns, ts_ns, ys_ns):

    stan_file = '%s/%s' % (STAN_FOLDER, 'basic_model_mix.stan')

    K = len(iter(x_ns).next())

    dim_d = {\
        'K':K,\
        }

    N = len(x_ns)
    assert len(x_ns) == len(ts_ns)
    assert len(ts_ns) == len(ys_ns)
    ls = np.array([len(ts) for ts in ts_ns])
    vs = np.concatenate(ys_ns)
    ts = np.concatenate(ts_ns)
    L = len(vs)
    assert len(vs) == len(ts)

    data_d = {
            'N':N,\
            'ls':ls,\
            'vs':vs,\
            'ts':ts,\
            'L':L,\
            'xs':x_ns,\
            'ss':s_ns,\
            }

    d = dict(dim_d.items() + data_d.items() + everything_dist_to_param_dict(everything_dist).items())

    fit = pystan.stan(file=stan_file, data=d, iter=n_steps, seed=random_seed, chains=num_chains)
    return traces(fit.extract(permuted=True), fit.extract(permuted=False), fit.data)

def get_everything_with_test_dist_traces_helper(n_steps, random_seed, num_chains, everything_dist, s_ns, x_ns, ts_ns, ys_ns, s_ns_test, x_ns_test):

    stan_file = '%s/%s' % (STAN_FOLDER, 'basic_model_mix_with_test.stan')

    K = len(iter(x_ns).next())
    
    dim_d = {\
        'K':K,\
        }

    N = len(x_ns)
    assert len(x_ns) == len(ts_ns)
    assert len(ts_ns) == len(ys_ns)
    K = len(iter(x_ns).next())
    ls = np.array([len(ts) for ts in ts_ns])
    vs = np.concatenate(ys_ns)
    ts = np.concatenate(ts_ns)
    L = len(vs)
    assert len(vs) == len(ts)

    data_d = {\
            'N':N,\
            'ls':ls,\
            'vs':vs,\
            'ts':ts,\
            'L':L,\
            'xs':x_ns,\
            'ss':s_ns,\
            }

    N_test = len(x_ns_test)
#    assert len(x_ns_test) == len(ts_ns_test)
#    assert len(ts_ns_test) == len(ys_ns_test)
#    ls_test = np.array([len(ts) for ts in ts_ns_test])
#    vs_test = np.concatenate(ys_ns_test)
#    ts_test = np.concatenate(ts_ns_test)
#    L_test = len(vs_test)
#    assert len(vs_test) == len(ts_test)

    data_d_test = {\
            'N_test':N_test,\
 #           'ls_test':ls_test,\
 #           'vs_test':vs_test,\
 #           'ts_test':ts_test,\
 #           'L_test':L_test,\
            'xs_test':x_ns_test,\
            'ss_test':s_ns_test,\
            }

    d = dict(dim_d.items() + data_d.items() + data_d_test.items() + everything_dist_to_param_dict(everything_dist).items())

    fit = pystan.stan(file=stan_file, data=d, iter=n_steps, seed=random_seed, chains=num_chains)
    return traces(fit.extract(permuted=True), fit.extract(permuted=False), fit.data)


def get_everything_with_test_phis_fixed_dist_traces_helper(n_steps, random_seed, num_chains, everything_dist, s_ns, x_ns, ts_ns, ys_ns, s_ns_test, x_ns_test):

    stan_file = '%s/%s' % (STAN_FOLDER, 'basic_model_mix_with_test_phis_fixed.stan')

    K = len(iter(x_ns).next())
    
    dim_d = {\
        'K':K,\
        }

    N = len(x_ns)
    assert len(x_ns) == len(ts_ns)
    assert len(ts_ns) == len(ys_ns)
    K = len(iter(x_ns).next())
    ls = np.array([len(ts) for ts in ts_ns])
    vs = np.concatenate(ys_ns)
    ts = np.concatenate(ts_ns)
    L = len(vs)
    assert len(vs) == len(ts)

    data_d = {\
            'N':N,\
            'ls':ls,\
            'vs':vs,\
            'ts':ts,\
            'L':L,\
            'xs':x_ns,\
            'ss':s_ns,\
            }

    N_test = len(x_ns_test)
#    assert len(x_ns_test) == len(ts_ns_test)
#    assert len(ts_ns_test) == len(ys_ns_test)
#    ls_test = np.array([len(ts) for ts in ts_ns_test])
#    vs_test = np.concatenate(ys_ns_test)
#    ts_test = np.concatenate(ts_ns_test)
#    L_test = len(vs_test)
#    assert len(vs_test) == len(ts_test)

    data_d_test = {\
            'N_test':N_test,\
 #           'ls_test':ls_test,\
 #           'vs_test':vs_test,\
 #           'ts_test':ts_test,\
 #           'L_test':L_test,\
            'xs_test':x_ns_test,\
            'ss_test':s_ns_test,\
            }

    param_d = {\
            'pop_a':everything_dist.abc_ns_dist.a_dist.pop_val,\
            'pop_b':everything_dist.abc_ns_dist.b_dist.pop_val,\
            'pop_c':everything_dist.abc_ns_dist.c_dist.pop_val,\
            's_a':everything_dist.B_phis_dist.s_a,\
            's_b':everything_dist.B_phis_dist.s_b,\
            's_c':everything_dist.B_phis_dist.s_c,\
            'phi_a':everything_dist.B_phis_dist.phi_a_dist.val,\
            'phi_b':everything_dist.B_phis_dist.phi_b_dist.val,\
            'phi_c':everything_dist.B_phis_dist.phi_c_dist.val,\
            'l_m':everything_dist.noise_dist_dist.l_m,\
            }
            
    d = dict(dim_d.items() + data_d.items() + data_d_test.items() + param_d.items())

    fit = pystan.stan(file=stan_file, data=d, iter=n_steps, seed=random_seed, chains=num_chains)
    return traces(fit.extract(permuted=True), fit.extract(permuted=False), fit.data)


def get_agreggate_loss(aggregate_f, loss_f, ts_ns, true_ys_ns, predicted_ys_ns):
    """
    assume times are ints
    """

def decay_f(initial, asymptotic, c, t):
    return initial - (asymptotic - initial) * np.exp(-t / c)
    
def fit_decay_f(ts, ys):
    import scipy.optimize
    def obj_f(x):
        initial, asymptotic, c = x
        return np.sum(np.square(np.array(map(functools.partial(decay_f, initial, asymptotic, c), ts)) - ys))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0.00,1.0),[0.00,1.0],[0.01,None]])
    return functools.partial(decay_f, *x)


def fit_average_shape(s_ns, ts_ns, ys_ns):
    d = pd.DataFrame([pd.Series(ys_i, index=ts_i) for (ts_i,ys_i) in itertools.izip(ts_ns,ys_ns)])
    
    def obj_f((a,b,c)):
        pred = pd.DataFrame([pd.Series([recovery_utils.f(s_i,a,b,c,t) for t in ts_i], index=ts_i) for (s_i,ts_i) in itertools.izip(s_ns,ts_ns)])
        ans = (pred - d).apply(lambda x: x**2).sum().sum()
        """
        idx=1
        t = 2
        print d.iloc[idx,:]
        print d.iloc[idx,t],pred.iloc[idx,t], a, b, c, s_ns[idx], recovery_utils.f(s_ns[idx],a,b,c,d.columns[t])
        print ans
        """
        return ans
    (a,b,c), f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array([0.5, 0.5, 2.0]), approx_grad = True, bounds = [(0.00,1.0),[0.00,1.0],[0.01,None]])

    print 'OOOOOOO', a, b, c
     
    return a,b,c

def abs_timewise_error(true_ys_ns, pred_ys_ns, ts_ns):
    """
    returns series
    """
    true_d = pd.DataFrame([pd.Series(ys_i, index=ts_i) for (ts_i,ys_i) in itertools.izip(ts_ns,true_ys_ns)])
    pred_d = pd.DataFrame([pd.Series(ys_i, index=ts_i) for (ts_i,ys_i) in itertools.izip(ts_ns,pred_ys_ns)])
    return (true_d - pred_d).apply(lambda s:s.apply(lambda x:abs(x))).apply(lambda s:s.mean())

class everything_dist_fitter(object):

    def __init__(self, B_phis_dist_fitter, noise_dist_dist):
        self.B_phis_dist_fitter, self.noise_dist_dist = B_phis_dist_fitter, noise_dist_dist

    def __call__(self, recovery_X_train, ys_ns_train):
        s_ns_train, x_ns_train, ts_ns_train = recovery_X_train
        pop_a, pop_b, pop_c = fit_average_shape(s_ns_train, ts_ns_train, ys_ns_train)
        _a_dist = a_dist(pop_a)
        _b_dist = b_dist(pop_b)
        _c_dist = c_dist(pop_c)
        _abc_ns_dist = abc_ns_dist(_a_dist, _b_dist, _c_dist)
        B_phis_dist = self.B_phis_dist_fitter.fit(recovery_X_train, ys_ns_train)
        return everything_dist(B_phis_dist, _abc_ns_dist, self.noise_dist_dist)
    
class recovery_curve(object):

    def __init__(self, s, a, b, c):
        self.s, self.a, self.b, self.c = s, a, b, c

    def __call__(self, t):
        return recovery_utils.f(self.s, self.a, self.b, self.c, t)

class recovery_predictor(object):
    """
    traces_helper has to be compatible with everything_dist
    """
    def __init__(self, n_steps, random_seed, everything_dist_fitter, traces_helper, point_f, num_chains, recovery_X_train, ys_ns_train):
        self.n_steps, self.random_seed, self.everything_dist_fitter, self.traces_helper, self.point_f, self.num_chains = n_steps, random_seed, everything_dist_fitter, traces_helper, point_f, num_chains
        self.recovery_X_train, self.ys_ns_train = recovery_X_train, ys_ns_train

    def predict(self, recovery_X_test):
        """
        returns 
        """
        s_ns_test, x_ns_test, ts_ns_test = recovery_X_test
        s_ns_train, x_ns_train, ts_ns_train = self.recovery_X_train
        everything_dist = self.everything_dist_fitter(self.recovery_X_train, self.ys_ns_train)
        traces = self.traces_helper(self.n_steps, self.random_seed, self.num_chains, everything_dist, s_ns_train, x_ns_train, ts_ns_train, self.ys_ns_train, s_ns_test, x_ns_test)
        recovery_curve_samples = [\
                                  [recovery_curve(s,a,b,c) for (a,b,c) in itertools.izip(a_i_samples, b_i_samples, c_i_samples)] \
                                  for (s, a_i_samples, b_i_samples, c_i_samples) in itertools.izip(s_ns_test, traces.param_to_trace('as_test').T, traces.param_to_trace('bs_test').T, traces.param_to_trace('cs_test').T)\
                                  ]
        self.traces = traces
        return np.array([\
                        np.array([self.point_f([rc_i_sample(t) for rc_i_sample in rc_i_samples]) for t in ts_i])\
                        for (rc_i_samples, ts_i) in itertools.izip(recovery_curve_samples,ts_ns_test)\
                        ])

    @property
    def train_info(self):
        raise NotImplementedError


class recovery_fitter(object):

    def __init__(self, n_steps, random_seed, num_chains, everything_dist_fitter, traces_helper, point_f):
        self.n_steps, self.random_seed, self.everything_dist_fitter, self.traces_helper, self.point_f, self.num_chains = n_steps, random_seed, everything_dist_fitter, traces_helper, point_f, num_chains

    def fit(self, recovery_X_train, ys_ns_train):
        return recovery_predictor(self.n_steps, self.random_seed, self.everything_dist_fitter, self.traces_helper, self.point_f, self.num_chains, recovery_X_train, ys_ns_train)
    

class mean_scaled_predictor(object):
    
    def __init__(self, t_to_scaled_val_d):
        self.t_to_scaled_val_d = t_to_scaled_val_d
    
    def predict(self, recovery_X_test):
        s_ns_test, x_ns_test, ts_ns_test = recovery_X_test
        return np.array([\
                [s_i*self.t_to_scaled_val_d[t] for t in ts_i]\
                for (ts_i, s_i) in itertools.izip(ts_ns_test, s_ns_test)\
                ])

        
class mean_scaled_fitter(object):

    def fit(self, recovery_X_train, ys_ns_train):
        s_ns_train, x_ns_train, ts_ns_train = recovery_X_train
        return mean_scaled_predictor(pd.DataFrame([pd.Series(ys_i/s_i, index=ts_i) for (s_i,ys_i,ts_i) in itertools.izip(s_ns_train, ys_ns_train, ts_ns_train)]).mean())


class logreg_predictor(object):

    def __init__(self, Bs):
        """
        Bs is a dataframe, time is columns
        """
        self.Bs = Bs

    def predict(self, recovery_X_test):
        s_ns_test, x_ns_test, ts_ns_test = recovery_X_test
        return np.array([\
                         s_i * np.array([recovery_utils.logistic(self.Bs[t].dot(x_i)) for t in ts_i])\
                         for (s_i,x_i,ts_i) in itertools.izip(s_ns_test,x_ns_test,ts_ns_test)\
                         ])


class logreg_fitter(object):

    def fit(self, recovery_X_train, ys_ns_train):

        import scipy.optimize
        
        s_ns_train, x_ns_train, ts_ns_train = recovery_X_train
        
        Bs = pd.DataFrame()
        d = pd.DataFrame([pd.Series(ys_i, index=ts_i) for (ts_i,ys_i) in itertools.izip(ts_ns_train, ys_ns_train)])
        all_idxs = d.index.values
        
        logistic = np.vectorize(recovery_utils.logistic)
        
        for (t,vals) in d.T.iterrows():

#            pdb.set_trace()
            
            def obj_f(B):
#                print logistic(x_ns_train.dot(B)) * s_ns_train
                return (pd.Series(logistic(x_ns_train.dot(B)) * s_ns_train, index=all_idxs) - vals).apply(lambda x:x**2).sum()

            B, f, d = scipy.optimize.fmin_l_bfgs_b(obj_f, np.array(np.zeros(x_ns_train.shape[1])), approx_grad = True)
            Bs[t] = B

        return logreg_predictor(Bs)

recovery_X_elt = namedtuple('recovery_X_elt', ['s','x','ts'])
        
    
recovery_X_base = namedtuple('recovery_X', ['s_ns', 'x_ns', 'ts_ns'])

class recovery_X(recovery_X_base):
    """
    s_ns, x_ns, ts_ns are all arrays, so that they support subsetting
    """

    def my_getitem(self, i):
        if isinstance(i,int):
            return recovery_X_elt(self.s_ns[i], self.x_ns[i], self.ts_ns[i])
        elif basic_utils.is_iterable(i) or isinstance(i,slice):
            return recovery_X(self.s_ns[i], self.x_ns[i], self.ts_ns[i])
        raise Exception

    def __len__(self):
        return len(self.s_ns)

    
class kfold_recovery_data_cv(object):

    def __init__(self, k):
        self.k = k

    def __call__(self, recovery_X, ys_ns):
        from sklearn.cross_validation import KFold
        kf = KFold(len(recovery_X), n_folds=self.k)
        return [((recovery_X.my_getitem(train_idx), ys_ns[train_idx]), (recovery_X.my_getitem(test_idx), ys_ns[test_idx])) for (train_idx,test_idx) in kf]

    
############################
# BELOW IS SCRATCH FOR NOW #
############################
    
recovery_datum = namedtuple('recovery_datum', ['s','x','ts','ys'])
        
recovery_data_base = namedtuple('recovery_data_base', ['s_ns','x_ns','ts_ns','ys_ns'])
    
class recovery_data(recovery_data_base):

    def __len__(self):
        return len(self.s_ns)

    def __iter__(self):
        return itertools.imap(recovery_datum, iter(self.s_ns), iter(self.x_ns), iter(self.ts_ns), iter(self.ys_ns))
    
#    def __getitem__(self, i):
#        return recovery_data(np.array(self.s_ns[i]), self.x_ns[i], self.ts_ns[i], self.ys_ns[i])
