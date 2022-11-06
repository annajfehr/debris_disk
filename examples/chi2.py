import sys
sys.path.append('/Volumes/disks/anna/arks/')
import debris_disk as DD
import time
import numpy as np

toc = time.time()

vis = DD.UVData('../49ceti_fits/')
#vis = DD.UVData('../../hd106906/mcmc_data/')

#print(set([vis.datasets[i].freq0 for i in range(2)]))

print('generating model')
obs_params = {'nu' : vis.datasets[0].freq0/1e9, # ghz
              'imres' : 0.03, # arcsec/pixel
              'distance' : 57.0} # parsecs


rad_params = {'alpha_in' : 1,
              'alpha_out' : -2,
              'Rin' : 20,
              'rc' : 200,
              'Rout' : 363,
              'gamma' : 5}
#rad_params = {'alpha' : 0.8,
#              'Rin' : 2 * DD.constants.AU,
#              'Rout' : 300 * DD.constants.AU}
#rad_params = {'rc' : 107* DD.constants.AU,
#              'alpha_in' : 1.75,
#              'alpha_out' : -1.47,
#              'gamma' : 1.}

mod  = DD.Disk(L_star=14.93, # L_sun
               inc=98, # Degrees
               sigma_crit = 1.24e-15,
               radial_func='double_powerlaw', # options: gaussian
               radial_params=rad_params, 
               vert_func='gaussian', 
               vert_params={'Hc' : 0.044,
                            'Rc' : 1,
                            'psi' : 1.},
               obs=obs_params)
#mod.shift(PA=-71.4)
mod.save('testing_for_casa.fits')
mod.square()
print('model generated')
print(vis.chi2(mod, PA=-74, dDec=0., dRA=0.))
vis.sample(mod, PA=-74, dDec=0., dRA=0.)
tic = time.time()

print("TIME = ", (tic - toc))
