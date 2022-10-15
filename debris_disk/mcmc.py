import time
import numpy as np
from debris_disk import profiles
from schwimmbad import MPIPool
from emcee import EnsembleSampler

class mcmc:
    def __init__(self,
                 uvdata=None, 
                 const_args=None, 
                 pranges=None, 
                 scale=None):
        self.locate_params(const_args)
        self.ndim = len(p0)
        self.p0 = p0
        self.scale=scale

    def locate_params(self, const_args):
        self.radial_func = const_args.pop('radial_func')
        self.gap = const_args.pop('gap')
        self.vert_func = const_args.pop('vert_func')
        
        if self.radial_func == 'powerlaw':
            self.radial_params = profiles.powerlaw.params()

        self.radial_params = {key : const_args.get(key, self.radial_params[key]) \
                for key in self.radial_params}

        if self.gap:
            self.gap_params = {'depth' : None,
                               'sigma' : None,
                               'x0' : None}
            self.gap_params = _gap_params(self.gap_params, const_args)
    
    
    def lnpost(p, chain):
        sys.stdout.flush()
        if not check_boundary: 
            return -np.inf

        return 1/(p[0] - 10) 

    def check_boundary(self, pos):
        for key in pos:
            if pos[key] < self.ranges[key][0] or pos[key] > self.ranges[key][1]:
                return False

        return True

    def unpack_params(self, pos):
        radial_params = {key : pos.get(key, self.radial_params[key]) \
                for key in self.radial_params}
        gap_params = _gap_params(self.gap_params, pos)

    def run(self, 
            nwalkers=10, 
            nsteps=100, 
            nthreads=1, 
            restart=None,
            outfile='mcmc.npy'):
        print("\nEmcee setup:")
        print("   Steps = " + str(nsteps))
        print("   Walkers = " + str(nwalkers))
        print("   Threads = " + str(nthreads))

        sampler = EnsembleSampler(nwalkers, self.ndim, mcmc.lnpost, args=[self])

        start = time.time()
        prob, state = None, None
        
        init_pos = np.random.normal(loc=self.p0,
                                    size=(nwalkers,self.ndim),
                                    scale=self.scale)
        run = sampler.sample(init_pos, iterations=nsteps, store=True)

        steps=[]
        
        with open(outfile, 'wb') as f:
            for i, (pos, lnprob, _) in enumerate(run):
                np.save(f, np.c_[pos, lnprob.T])
                #new_step=[np.append(pos2[k], lnprobs[k]) for k in range(nwalkers)]
                #np.save(f, lnpro
    
def _gap_params(gap_params, args):
        gap_params['depth'] = args.get('gap_depth', \
                self.gap_params[depth])
        gap_params['sigma'] = args.get('gap_sigma', \
                self.gap_params[depth])
        gap_params['x0'] = args.get('gap_x0', \
                self.gap_params[depth])
        return gap_params
