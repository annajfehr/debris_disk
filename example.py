import debris_disk as DD
import time
import numpy as np

toc = time.time()


vis = DD.UVData('../hd106906/mcmc_data/')
    
obs_params = {'nu' : 345.8, # ghz
              'imres' : 0.005, # arcsec/pixel
              'PA' : 128, # degrees
              'distance' : 100} # parsecs

mod  = DD.Disk(Lstar=1., # L_sun
               Mdust=1e-7, # Mdust
               inc=88.5, # Degrees
               radial_func='powerlaw', # options: gaussian
               radial_params=[2.2, 
                              22 * DD.constants.AU, 
                              42 * DD.constants.AU], 
               aspect_ratio = 0.025,
               vert_func='gaussian', 
               obs=obs_params)


mod.square()
mod.rotate()
mod.save('example.fits')

vis.chi2(mod)

tic = time.time()

print("TIME = ", (tic - toc))
