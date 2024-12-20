import json
import math
import random
import os
import sys
import time
import numpy as np
from debris_disk import profiles
from debris_disk import Disk
import debris_disk as DD
from schwimmbad import MPIPool
from emcee import EnsembleSampler

class MCMC:
    """
    A class that holds an mcmc chain.

    ...
    
    Attributes
    ----------
    uvdata : string
        Filepath of uv fits directory
    obs_params : dict
        Dictionary of observation parameters
            nu : frequency in ghz
            imres : model resolution in " / pixels
            distance : distance in pc
    fixed_args : dict
        Dictionary of fixed parameters
        Should include
            radial_func
            vert_func
        and disk model parameters not being fitted
    p0 : dict
        Initial guesses for parameters to fit
    pranges : dict
        
    """
    def __init__(self,
                 uvdata, # data_fps_list to be turned into UVData object
                 obs_params,
                 fixed_args, # from param json
                 p0, # from param json
                 pranges, # from param json
                 name=None,
                 filetype='txt',
                 pscale=None): # from param json
        self.uvdata=uvdata
        vis = DD.UVDataset(uvdata, filetype=filetype)
        self.obs_params = DD.Observation(vis=vis,
                                         json_file='/arc/projects/ARKS/parametric_modeling/REASONS.json',
                                         #json_file='/Volumes/disks/brianna/REASONS.json', # for kazul/katsa versions
                                         #json_file='/zfshomes/bzawadzki/REASONS.json', # for Wes cluster version
                                         sys_name=name)
        self.vis = vis.__dict__
        self.filetype=filetype
        
        self.parse_fixed_params(fixed_args)
        self.param_dict_to_list(p0, pscale, pranges)

    def parse_fixed_params(self, fixed_args):
        """Place fixed_args into self.fixed_rad_params, self.fixed_vert_params, 
        and self.fixed_args.

        Parameters
        ----------
        fixed_args : dict
            Dictionary of fixed arguments

        Returns
        -------
        None
        """

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
        """Place initial parameter guesses, scale, and ranges into radial,
        vertical, and other dictionaries.

        Parameters
        ----------
        p0 : dict
            Initial guesses for free parameters
        scale : dict
            Scatter for initial walker positions
        ranges : dict
            Limits on walker position

        The keys for p0, scale, and ranges must all be identical, as in
        subdictionaries

        Returns
        -------
        None
        """

        assert set(p0.keys()) == set(ranges.keys())
        assert set(p0.keys()) == set(scale.keys())
        
        self.params=[]
        self.p0=[]
        self.scale=[]
        self.ranges=[]

        try:
            rad_p0 = p0.pop('radial_params')
            rad_ranges = ranges.pop('radial_params')
            rad_scale = scale.pop('radial_params')

            assert rad_p0.keys() == rad_ranges.keys()
            assert rad_p0.keys() == rad_scale.keys()
            
            self.params += list(rad_p0.keys())
            self.p0 += list(rad_p0.values())
            self.scale += list(rad_scale.values())
            self.ranges += list(rad_ranges.values())
            self.num_rad = len(rad_p0)
        except:
            self.num_rad = 0
        
        try:
            vert_p0 = p0.pop('vert_params')
            vert_ranges = ranges.pop('vert_params')
            vert_scale = scale.pop('vert_params')
            
            assert vert_p0.keys() == vert_ranges.keys()
            assert vert_p0.keys() == vert_scale.keys()

            self.params += list(vert_p0.keys())
            self.p0 += list(vert_p0.values())
            self.scale += list(vert_scale.values())
            self.ranges += list(vert_ranges.values())
            self.num_vert = len(vert_p0)
        except:
            self.num_vert = 0

        self.params += list(p0.keys())
        self.p0 += list(p0.values())
        self.scale += list(scale.values())
        self.ranges += list(ranges.values())
        
        self.ndim = len(self.p0)

    def run(self, 
            nwalkers=10, 
            nsteps=100, 
            parallel=False, 
            restart=None,
            outfile='mcmc.txt',
            mode='old',
            file_dir='.',
            verbose=False):
        print("\nEmcee setup:")
        print("   Steps = " + str(nsteps))
        print("   Walkers = " + str(nwalkers))


        nwalkers = int(nwalkers)
        nsteps = int(nsteps)


        start = time.time()
        prob, state = None, None
        

        steps=[]

        assert os.path.exists(file_dir)

        if parallel:
            pool=MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        else:
            pool=None

        if mode=='old':
            if type(self.obs_params) == dict:
                while not np.isscalar(self.obs_params['nu']):
                    self.obs_params['nu'] = self.obs_params['nu'][0]
                self.obs_params['nu'] = np.array([self.obs_params['nu']])
            else:
                while not np.isscalar(self.obs_params.nu):
                    self.obs_params.nu = self.obs_params.nu[0]
                self.obs_params.nu = np.array([self.obs_params.nu])

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
                                        self.uvdata,
                                        self.vis,
                                        self.filetype,
                                        mode,
                                        file_dir,
                                        verbose], 
                                  pool=pool)
        
        if os.path.exists(outfile):
            restart=outfile

        if restart:
            steps = np.loadtxt(restart)
            init_pos = np.array([steps[-(nwalkers-i+1)] for i in range(nwalkers)])[:,:-1]
        else:
            init_pos = np.random.normal(loc=self.p0,
                                        size=(nwalkers,self.ndim),
                                        scale=self.scale)
        if verbose:
            print("init pos!!")
            print(init_pos)
        run = sampler.sample(init_pos, iterations=nsteps, store=True)

        print('Beginning Chain')

        for i, (pos, lnprob, _) in enumerate(run):
            if verbose:
                print("\n")
                print("run information, i=", str(i))
                print("param vals: ")
                print(pos)
                print("lnprobs for a step: ")
                print(lnprob)
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

    viewing_params = {'PA' : params_dict.pop('PA'), 'dRA' : params_dict.pop('dRA'), 'dDec' : params_dict.pop('dDec'), 'F_star' : params_dict.pop('F_star')}

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
           uvdata,
           uvdict,
           filetype,
           mode,
           file_dir,
           verbose):
    sys.stdout.flush()
    if not check_boundary(ranges, p): 
        if verbose:
            print('Parameters out of bounds')
        return -np.inf
    disk_params, viewing_params = param_list_to_dict(p, 
                                                     params, 
                                                     num_rad,
                                                     num_vert,
                                                     fixed_rad_params,
                                                     fixed_vert_params)
    disk_params['sigma_crit'] = 10 ** disk_params['sigma_crit']
    if verbose:
        print('DISK_PARAM = ', disk_params)
        print('VIEW_PARAM = ', viewing_params)
    mod = Disk(obs=obs_params, F_star=viewing_params['F_star'], **fixed_args, **disk_params)
    
    if verbose:
        print('Generated model')

    if mode=='new':
        try:
            # create a UVDataset from:
            # - the text files we were using
            # - the UVData instances we were using
            # - the mrs and resolutions (in radials)
            # - chans (in lambda [m])
            # - freq (a single frequency as of 5/22/24)
            vis = DD.UVDataset(stored=uvdict)
            # now get the model disk (initialized above) and compute chi2
            chi2 = vis.chi2(disk=mod, **viewing_params)
        except:
            if verbose:
                print('Model not evaluated')
            return -np.inf
    if mode=='old':
        try:
            key = str(np.random.randint(99999))
            mod.save(file_dir+key+'mcmc')
            del mod
            chi2 = DD.uvdata.chiSq(uvdata, file_dir+key+'mcmc0', filetype=filetype, **viewing_params)
            os.remove(file_dir+key+'mcmc0.fits')
        except:
            if os.path.exists(file_dir+key+'mcmc0.fits'):
                os.remove(file_dir+key+'mcmc0.fits')

            if verbose:
                print('Model not evaluated')
            return -np.inf

    if math.isnan(chi2):
        return -np.inf

    if verbose:
        print(-0.5 * chi2)
    return -0.5 * chi2
