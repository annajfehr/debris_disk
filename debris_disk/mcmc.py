import time
import sys
import os
import numpy as np
from debris_disk import profiles
from debris_disk import Disk
import debris_disk as DD
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


        sampler = EnsembleSampler(nwalkers, 
                                  self.ndim, 
                                  lnpost, 
                                  args=[self.params,
                                        self.ranges,
                                        self.num_rad,
                                        self.num_vert,
                                        self.fixed_rad_params,
                                        self.fixed_vert_params,
                                        self.obs_params,
                                        self.fixed_args,
                                        self.uvdata], 
                                  pool=pool)

        run = sampler.sample(init_pos, iterations=nsteps, store=True)

        print('Beginning Chain')

        for i, (pos, lnprob, _) in enumerate(run):
            with open(outfile, 'a+') as f:
                np.savetxt(f, np.c_[pos, lnprob.T])



def param_list_to_dict(pos, 
                       params,
                       num_rad,
                       num_vert, 
                       fixed_rad_params,
                       fixed_vert_params):

    rad_params = params[:num_rad]
    rad_vals = pos[:num_rad]
    radial_dict = {**fixed_rad_params,
                   **dict(zip(rad_params, rad_vals))}
    
    vert_params = params[num_rad:num_rad+num_vert]
    vert_vals = pos[num_rad:num_rad+num_vert]
    vert_dict = {**fixed_vert_params,
                 **dict(zip(vert_params, vert_vals))}
    
    params = params[num_rad+num_vert:]
    vals = pos[num_rad+num_vert:]
    params_dict = dict(zip(params, vals))

    params_dict['radial_params'] = radial_dict
    params_dict['vert_params'] = vert_dict

    viewing_params = {'PA' : params_dict.pop('PA')}

    return params_dict, viewing_params

def check_boundary(ranges, pos):
    """
    Check if any parameters are out of bounds
    """
    for i, p in enumerate(pos):
        if p < ranges[i][0] or p > ranges[i][1]:
            return False

    return True

def lnpost(p,
           params,
           ranges,
           num_rad,
           num_vert, 
           fixed_rad_params,
           fixed_vert_params,
           obs_params,
           fixed_args,
           uvdata):
    sys.stdout.flush()
    if not check_boundary(ranges, p): 
        return -np.inf

    disk_params, viewing_params = param_list_to_dict(p, 
                                                     params, 
                                                     num_rad,
                                                     num_vert,
                                                     fixed_rad_params,
                                                     fixed_vert_params)
    
    mod = Disk(obs=obs_params, **fixed_args, **disk_params)
    vis = DD.UVData(uvdata, mode='mcmc')
    chi2 = vis.chi2(mod, **viewing_params)
    return -0.5 * chi2
