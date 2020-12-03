import sys
import high_dimensional_sampling as hds
try:
    import pygmo as pg  # noqa: F401
    from pygmo import *  # noqa: F403, F401
except ImportError:
    pass
import numpy as np

# based on https://esa.github.io/pagmo2

# All the global optimization algorithms in the webpage list
# (https://esa.github.io/pagmo2/docs/algorithm_list.html) have been
# included here but some are not recognized probably because there is not an
# active python backend for them and others are meant for things like
# multiobjetive optimation, etc ... and threfore are not actived.
# The effective list is below.

algorithms = [
    'sade', 'gaco', 'gwo', 'bee_colony', 'de', 'sea', 'sga', 'de1220', 'cmaes',
    'compass_search', 'simulated_annealing', 'pso', 'pso_gen', 'mbh',
    'cstrs_self_adaptive'
]


class Pygmo(hds.Procedure):
    def __init__(self,
                 scanner="sade",
                 gen=100,
                 variant=2,
                 allowed_variants=[2, 3, 7, 10, 13, 14, 15, 16],
                 variant_adptv=1,
                 ftol=1e-6,
                 xtol=1e-6,
                 ker=63,
                 q=1.0,
                 oracle=0.,
                 acc=0.01,
                 threshold=1,
                 n_gen_mark=7,
                 impstop=100000,
                 evalstop=100000,
                 focus=0.,
                 F=0.8,
                 CR=0.9,
                 memory=False,
                 cr=0.9,
                 eta_c=1,
                 m=0.02,
                 param_m=1,
                 param_s=2,
                 crossover='exponential',
                 mutation='polynomial',
                 selection='tournament',
                 cc=-1,
                 cs=-1,
                 c1=-1,
                 cmu=-1,
                 sigma0=0.5,
                 force_bounds=True,
                 weight_generation="grid",
                 decomposition="tchebycheff",
                 neighbours=20,
                 eta_m=20,
                 realb=0.9,
                 limit=2,
                 preserve_diversity=True,
                 max_fevals=1,
                 start_range=.1,
                 stop_range=1.e-6,
                 reduction_coeff=.5,
                 Ts=10.,
                 Tf=.1,
                 n_T_adj=10,
                 n_range_adj=10,
                 bin_size=10,
                 omega=0.7298,
                 eta1=2.05,
                 eta2=2.05,
                 max_vel=0.5,
                 neighb_type=2,
                 neighb_param=4,
                 c2=0.5,
                 chi=0.5,
                 v_coeff=0.5,
                 leader_selection_range=2,
                 diversity_mechanism="crowding distance",
                 algo='None',
                 stop=5,
                 perturb=0.01,
                 iters=1,
                 phmcr=0.85,
                 ppar_min=0.35,
                 ppar_max=0.99,
                 bw_min=1e-5,
                 bw_max=1.,
                 eta_mu=-1,
                 eta_sigma=-1,
                 eta_b=-1,
                 seed=123456,
                 size=20,
                 log_data=False,
                 verbose=0):

        try:
            pg
        except NameError:
            raise ImportError(
                "The `pygmo` package is not installed. See the wiki on our "
                "GitHub project for installation instructions.")

        self.store_parameters = [
            'scanner', 'gen', 'variant', 'allowed_variants', 'variant_adptv',
            'ftol', 'xtol', 'ker', 'q', 'oracle', 'acc', 'threshold',
            'n_gen_mark', 'impstop', 'evalstop', 'focus', 'F', 'CR', 'memory',
            'cr', 'eta_c', 'm', 'param_m', 'param_s', 'crossover', 'mutation',
            'selection', 'cc', 'cs', 'c1', 'cmu', 'sigma0', 'force_bounds',
            'weight_generation', 'decomposition', 'neighbours', 'eta_m',
            'realb', 'limit', 'preserve_diversity', 'max_fevals',
            'start_range', 'stop_range', 'reduction_coeff', 'Ts', 'Tf',
            'n_T_adj', 'n_range_adj', 'bin_size', 'omega', 'eta1', 'eta2',
            'max_vel', 'neighb_type', 'neighb_param', 'c2', 'chi', 'v_coeff',
            'leader_selection_range', 'diversity_mechanism', 'algo', 'stop',
            'perturb', 'iters', 'phmcr', 'ppar_min', 'ppar_max', 'bw_min',
            'bw_max', 'eta_mu', 'eta_sigma', 'eta_b', 'seed', 'size'
        ]

        # Pass pagmo algoritm
        self.scanner = scanner
        self.gen = gen
        self.variant = variant
        self.allowed_variants = allowed_variants
        self.variant_adptv = variant_adptv
        self.ftol = ftol
        self.xtol = xtol
        self.ker = ker
        self.q = q
        self.oracle = oracle
        self.acc = acc
        self.threshold = threshold
        self.n_gen_mark = n_gen_mark
        self.impstop = impstop
        self.evalstop = evalstop
        self.focus = focus
        self.F = F
        self.CR = CR
        self.memory = memory
        self.cr = cr
        self.eta_c = eta_c
        self.m = m
        self.param_m = param_m
        self.param_s = param_s
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.cc = cc
        self.cs = cs
        self.c1 = c1
        self.cmu = cmu
        self.sigma0 = sigma0
        self.force_bounds = force_bounds
        self.weight_generation = weight_generation
        self.decomposition = decomposition
        self.neighbours = neighbours
        self.eta_m = eta_m
        self.realb = realb
        self.limit = limit
        self.preserve_diversity = preserve_diversity
        self.max_fevals = max_fevals
        self.start_range = start_range
        self.stop_range = stop_range
        self.reduction_coeff = reduction_coeff
        self.Ts = Ts
        self.Tf = Tf
        self.n_T_adj = n_T_adj
        self.n_range_adj = n_range_adj
        self.bin_size = bin_size
        self.start_range = start_range
        self.omega = omega
        self.eta1 = eta1
        self.eta2 = eta2
        self.max_vel = max_vel
        self.variant = variant
        self.neighb_type = neighb_type
        self.neighb_param = neighb_param
        self.c2 = c2
        self.chi = chi
        self.v_coeff = v_coeff
        self.leader_selection_range = leader_selection_range
        self.diversity_mechanism = diversity_mechanism
        self.algo = algo
        self.stop = stop
        self.perturb = perturb
        self.iters = iters
        self.phmcr = phmcr
        self.ppar_min = ppar_min
        self.ppar_max = ppar_max
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.eta_mu = eta_mu
        self.eta_sigma = eta_sigma
        self.eta_b = eta_b
        self.seed = seed
        self.size = size
        self.log_data = log_data
        self.verbose = verbose
        self.reset()

    def __call__(self, function):

        scanner_options = {
            'sade':
            dict(gen=self.gen,
                 variant=self.variant,
                 variant_adptv=self.variant_adptv,
                 ftol=self.ftol,
                 xtol=self.xtol,
                 memory=self.memory,
                 seed=self.seed),
            'gaco':
            dict(gen=self.gen,
                 ker=self.ker,
                 q=self.q,
                 oracle=self.oracle,
                 acc=self.acc,
                 threshold=self.threshold,
                 n_gen_mark=self.n_gen_mark,
                 impstop=self.impstop,
                 evalstop=self.evalstop,
                 focus=self.focus,
                 memory=self.memory,
                 seed=self.seed),
            'maco':
            dict(gen=self.gen,
                 ker=self.ker,
                 q=self.q,
                 threshold=self.threshold,
                 n_gen_mark=self.n_gen_mark,
                 evalstop=self.evalstop,
                 focus=self.focus,
                 memory=self.memory,
                 seed=self.seed),
            'gwo':
            dict(gen=self.gen, seed=self.seed),
            'bee_colony':
            dict(gen=self.gen, limit=self.limit, seed=self.seed),
            'de':
            dict(gen=self.gen,
                 F=self.F,
                 CR=self.CR,
                 variant=self.variant,
                 ftol=self.ftol,
                 xtol=self.xtol,
                 seed=self.seed),
            'sea':
            dict(gen=self.gen, seed=self.seed),
            'sga':
            dict(gen=self.gen,
                 cr=self.cr,
                 eta_c=self.eta_c,
                 m=self.m,
                 param_m=self.param_m,
                 param_s=self.param_s,
                 crossover=self.crossover,
                 mutation=self.mutation,
                 selection=self.selection,
                 seed=self.seed),
            'de1220':
            dict(gen=self.gen,
                 allowed_variants=self.allowed_variants,
                 variant_adptv=self.variant_adptv,
                 ftol=self.ftol,
                 xtol=self.xtol,
                 memory=self.memory,
                 seed=self.seed),
            'cmaes':
            dict(gen=self.gen,
                 cc=self.cc,
                 cs=self.cs,
                 c1=self.c1,
                 cmu=self.cmu,
                 sigma0=self.sigma0,
                 ftol=self.ftol,
                 xtol=self.xtol,
                 memory=self.memory,
                 force_bounds=self.force_bounds,
                 seed=self.seed),
            'moead':
            dict(gen=self.gen,
                 weight_generation=self.weight_generation,
                 decomposition=self.decomposition,
                 neighbours=self.neighbours,
                 CR=self.CR,
                 F=self.F,
                 eta_m=self.eta_m,
                 realb=self.realb,
                 limit=self.limit,
                 preserve_diversity=self.preserve_diversity,
                 seed=self.seed),
            'compass_search':
            dict(max_fevals=self.max_fevals,
                 start_range=self.start_range,
                 stop_range=self.stop_range,
                 reduction_coeff=self.reduction_coeff),
            'simulated_annealing':
            dict(Ts=self.Ts,
                 Tf=self.Tf,
                 n_T_adj=self.n_T_adj,
                 n_range_adj=self.n_range_adj,
                 bin_size=self.bin_size,
                 start_range=self.start_range,
                 seed=self.seed),
            'pso':
            dict(gen=self.gen,
                 omega=self.omega,
                 eta1=self.eta1,
                 eta2=self.eta2,
                 max_vel=self.max_vel,
                 variant=self.variant,
                 neighb_type=self.neighb_type,
                 neighb_param=self.neighb_param,
                 memory=self.memory,
                 seed=self.seed),
            'pso_gen':
            dict(gen=self.gen,
                 omega=self.omega,
                 eta1=self.eta1,
                 eta2=self.eta2,
                 max_vel=self.max_vel,
                 variant=self.variant,
                 neighb_type=self.neighb_type,
                 neighb_param=self.neighb_param,
                 memory=self.memory,
                 seed=self.seed),
            'nsga2':
            dict(gen=self.gen,
                 cr=self.cr,
                 eta_c=self.eta_c,
                 m=self.m,
                 eta_m=self.eta_m,
                 seed=self.seed),
            'nspso':
            dict(gen=self.gen,
                 omega=self.omega,
                 c1=self.c1,
                 c2=self.c2,
                 chi=self.chi,
                 v_coeff=self.v_coeff,
                 leader_selection_range=self.leader_selection_range,
                 diversity_mechanism=self.diversity_mechanism,
                 memory=self.memory,
                 seed=self.seed),
            'mbh':
            dict(algo=self.algo,
                 stop=self.stop,
                 perturb=self.perturb,
                 seed=self.seed),
            'cstrs_self_adaptive':
            dict(iters=self.iters, algo=self.algo, seed=self.seed),
            'ihs':
            dict(gen=self.gen,
                 phmcr=self.phmcr,
                 ppar_min=self.ppar_min,
                 ppar_max=self.ppar_max,
                 bw_min=self.bw_min,
                 bw_max=self.bw_max,
                 seed=self.seed),
            'xnes':
            dict(gen=self.gen,
                 eta_mu=self.eta_mu,
                 eta_sigma=self.eta_sigma,
                 eta_b=self.eta_b,
                 sigma0=self.sigma0,
                 ftol=self.ftol,
                 xtol=self.xtol,
                 memory=self.memory,
                 force_bounds=self.force_bounds,
                 seed=self.seed)
        }

        if self.log_data:
            xl = []
            yl = []

        log_data = self.log_data

        #
        class interf_function:
            def __init__(self, dim):
                self.dim = dim

            def fitness(self, x):
                x = np.expand_dims(x, axis=0)
                y = function(x)
                # x = x[0]
                y = y.tolist()
                if log_data:
                    xl.append(x)
                    yl.append(y)
                # print (x, y[0])
                return y[0]

            if function.is_differentiable():

                def gradient(self, x):
                    x = np.expand_dims(x, axis=0)
                    g = function(x)
                    g = g.tolist()
                    return g[0]

            def get_bounds(self):
                lb = []
                ub = []
                bounds = function.get_ranges()
                # warning
                # check for infinities
                for i in range(len(bounds)):
                    lb.append(bounds[i, 0])
                    ub.append(bounds[i, 1])
                r = (np.array(lb), np.array(ub))
                return r

        # I need to call pygmo functions directly
        prob = pg.problem(interf_function(function))

        # print (prob.get_thread_safety())

        if self.scanner == "sade":
            # I need a dictionary with algorithms and options
            algo = pg.algorithm(pg.sade(**scanner_options[self.scanner]))
        elif self.scanner == "gaco":
            algo = pg.algorithm(pg.gaco(**scanner_options[self.scanner]))
        # elif self.scanner == "maco": # is not implemented though in webpage
        #                               looks it is
        # algo = pg.algorithm(pg.maco(**scanner_options[self.scanner]))
        elif self.scanner == "gwo":
            algo = pg.algorithm(pg.gwo(**scanner_options[self.scanner]))
        elif self.scanner == "bee_colony":
            algo = pg.algorithm(pg.bee_colony(**scanner_options[self.scanner]))
        elif self.scanner == "de":
            algo = pg.algorithm(pg.de(**scanner_options[self.scanner]))
        elif self.scanner == "sea":
            algo = pg.algorithm(pg.sea(**scanner_options[self.scanner]))
        elif self.scanner == "sga":
            algo = pg.algorithm(pg.sga(**scanner_options[self.scanner]))
        elif self.scanner == "de1220":
            algo = pg.algorithm(pg.de1220(**scanner_options[self.scanner]))
        elif self.scanner == "cmaes":
            algo = pg.algorithm(pg.cmaes(**scanner_options[self.scanner]))
        # elif self.scanner == "moead": #multiobjective algorithm
        #  algo = pg.algorithm(pg.moead(**scanner_options[self.scanner]))
        elif self.scanner == "compass_search":
            algo = pg.algorithm(
                pg.compass_search(**scanner_options[self.scanner]))
        elif self.scanner == 'simulated_annealing':
            algo = pg.algorithm(
                pg.simulated_annealing(**scanner_options[self.scanner]))
        elif self.scanner == 'pso':
            algo = pg.algorithm(pg.pso(**scanner_options[self.scanner]))
        elif self.scanner == 'pso_gen':
            algo = pg.algorithm(pg.pso_gen(**scanner_options[self.scanner]))
        # elif self.scanner == 'nsga2': #multiobjective algorithm
        #  algo = pg.algorithm(pg.nsga2(**scanner_options[self.scanner]))
        # elif self.scanner == 'nspso': is not implemented though in webpage
        #                               looks it is
        #  algo = pg.algorithm(pg.nspso(**scanner_options[self.scanner]))
        elif self.scanner == 'mbh':
            if scanner_options[self.scanner]['algo'] == 'de':
                algo = pg.algorithm(
                    pg.mbh(pg.algorithm(pg.de(**scanner_options['de']))))
        # elif self.scanner == 'ihs': #does not work
        #  algo = pg.algorithm(ihs(**scanner_options[self.scanner]))
        # elif self.scanner == 'xnes': #does not work
        #  algo = pg.algorithm(xnes(**scanner_options[self.scanner]))
        # uda = algo.extract(xnes)
        else:
            print(
                'The ' + self.scanner + ' algorithm is not implemented. The '
                'list of algorithms available is', algorithms)
            sys.exit()

        # add verbosing flag
        if self.verbose > 1:
            algo.set_verbosity(self.verbose)

        pop = pg.population(prob, self.size)

        if self.verbose > 9:
            print('prob', prob)

        opt = algo.evolve(pop)

        if self.verbose > 9:
            print('algo', algo)

        # best_x = np.expand_dims(opt.champion_x, axis=0)
        # best_fitness = np.expand_dims(opt.get_f()[opt.best_idx()], axis=0)
        best_x = np.expand_dims(opt.champion_x, axis=0)
        best_fitness = np.expand_dims(opt.champion_f, axis=0)

        if self.verbose > 0:
            print('best fit:', best_x, best_fitness)

        if self.log_data:
            x = np.squeeze(xl, axis=(1, ))
            y = np.squeeze(yl, axis=(2, ))

        if self.log_data:
            return (x, y)
        else:
            return (best_x, best_fitness)

    def reset(self):
        self.current_position = None
        self.current_value = None

    def is_finished(self):
        return True

    def check_testfunction(self, function):
        return True
