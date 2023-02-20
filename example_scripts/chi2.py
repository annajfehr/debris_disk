import sys
sys.path.append('/arc/projects/ARKS/vertical_structure/parametric_modeling/')
import debris_disk as DD
import time
import numpy as np


vis = DD.UVDataset('../TYC9340/12m.txt', mode='mcmc')
freqs = DD.constants.c / vis.chans

toc = time.time()
obs_params = {'nu' : freqs, # ghz
              'imres' : 0.03, # arcsec/pixel
              'distance' : 36.6} # parsecs


rad_params = {'alpha_in' : 1,
              'alpha_out' : -2,
              'Rin' : 30,
              'rc' : 100,
              'Rout' : 130,
              'gamma' : 5}

mod  = DD.Disk(L_star=0.1866, # L_sun
               inc=40, # Degrees
               sigma_crit = 6.2e-15,
               radial_func='double_powerlaw', # options: gaussian
               radial_params=rad_params, 
               vert_func='gaussian', 
               vert_params={'Hc' : 0.044,
                            'Rc' : 1,
                            'psi' : 1.},
               obs=obs_params)

mod.save('testing_for_casa')
mod.square()
print(vis.chi2(disk=mod, PA=130, dDec=0., dRA=0.))
print('model generated')
vis.sample(disk=mod)
#DD.UVDataset.make_ms('TYC9340.12m.cal.bary.continuum.fav.tav.SMGsub.corrected.ms', 'mod.txt', mod_msfile='mod.ms', resid_msfile='resid.ms')

tic = time.time()

print("TIME = ", (tic - toc))
