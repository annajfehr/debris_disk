import time
import sys
import os
import numpy as np
from debris_disk import profiles
from debris_disk import Disk
from schwimmbad import MPIPool
from emcee import EnsembleSampler

class mcmc:
    def __init__(self,
                 uvdata, 
                 obs_params,
                 fixed_args,
                 p0,
                 pranges,
                 pscale=None):
        self.uvdata=uvdata
        self.obs_params=obs_params
        self.parse_fixed_params(fixed_args)
        self.param_dict_to_list(p0, pscale, pranges)

    def parse_fixed_params(self, fixed_args):
        try:
            self.fixed_rad_params = fixed_args.pop('radial_params')
        except:
            self.fixed_rad_params = {}
        
        try:
            self.fixed_vert_params = fixed_args.pop('vert_params')
        except:
            self.fixed_vert_params = {}

        self.fixed_args = fixed_args
        
    def param_dict_to_list(self, p0, scale, ranges):
        self.params=[]
        self.p0=[]
        self.scale=[]
        self.ranges=[]

        try:
            rad_p0 = p0.pop('radial_params')
            rad_scale = scale.pop('radial_params')
            rad_ranges = ranges.pop('radial_params')
            self.params += list(rad_p0.keys())
            self.p0 += list(rad_p0.values())
            self.scale += list(rad_scale.values())
            self.ranges += list(rad_ranges.values())
            self.num_rad = len(rad_p0)
        except:
            self.num_rad = 0.
        
        try:
            vert_p0 = p0.pop('vert_params')
            vert_scale = scale.pop('vert_params')
            vert_ranges = ranges.pop('vert_params')
            self.params += list(vert_p0.keys())
            self.p0 += list(vert_p0.values())
            self.scale += list(vert_scale.values())
            self.ranges += list(vert_ranges.values())
            self.num_vert = len(vert_p0)
        except:
            self.num_vert = 0.


        self.params += list(p0.keys())
        self.p0 += list(p0.values())
        self.scale += list(scale.values())
        self.ranges += list(ranges.values())
        
        self.ndim = len(self.p0)
        assert self.ndim == len(self.ranges)
        assert self.ndim == len(self.scale)

    def run(self, 
            nwalkers=10, 
            nsteps=100, 
            nthreads=1, 
            restart=None,
            outfile='mcmc.txt'):
        print("\nEmcee setup:")
        print("   Steps = " + str(nsteps))
        print("   Walkers = " + str(nwalkers))
        print("   Threads = " + str(nthreads))


        nwalkers = int(nwalkers)
        nsteps = int(nsteps)
        nthreads = int(nthreads)


        start = time.time()
        prob, state = None, None
        
        init_pos = np.random.normal(loc=self.p0,
                                    size=(nwalkers,self.ndim),
                                    scale=self.scale)

        steps=[]

        if nthreads > 1:
            pool=MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        else:
            pool=None

        sampler = EnsembleSampler(nwalkers, self.ndim, mcmc.lnpost, args=[self], pool=pool)
        run = sampler.sample(init_pos, iterations=nsteps, store=True)
        
        for i, (pos, lnprob, _) in enumerate(run):
            with open(outfile, 'a+') as f:
                np.savetxt(f, np.c_[pos, lnprob.T])

    def lnpost(p, chain):
        sys.stdout.flush()
        if not chain.check_boundary(p): 
            return -np.inf

        disk_params, viewing_params = chain.param_list_to_dict(p)
        mod = Disk(obs=chain.obs_params, **chain.fixed_args, **disk_params)
        chi2 = chain.uvdata.chi2(mod, **viewing_params)
        return -0.5 * chi2

    def check_boundary(self, pos):
        """
        Check if any parameters are out of bounds
        """
        for i, p in enumerate(pos):
            if p < self.ranges[i][0] or p > self.ranges[i][1]:
                return False

        return True

    def param_list_to_dict(self, pos):

        rad_params = self.params[:self.num_rad]
        rad_vals = pos[:self.num_rad]
        radial_dict = {**self.fixed_rad_params,
                       **dict(zip(rad_params, rad_vals))}
        
        vert_params = self.params[self.num_rad:self.num_rad+self.num_vert]
        vert_vals = pos[:self.num_rad:self.num_rad+self.num_vert]
        vert_dict = {**self.fixed_vert_params,
                     **dict(zip(vert_params, vert_vals))}
        
        params = self.params[self.num_rad+self.num_vert:]
        vals = pos[self.num_rad+self.num_vert:]
        params_dict = dict(zip(params, vals))

        params_dict['radial_params'] = radial_dict
        params_dict['vert_params'] = vert_dict

        viewing_params = {'PA' : params_dict.pop('PA')}

        return params_dict, viewing_params

