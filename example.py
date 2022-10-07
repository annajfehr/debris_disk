import debris_disk as DD
import time
import numpy as np

toc = time.time()


vis = DD.UVData('../hd106906/mcmc_data/', mode='MCMC')
    
obs_params = {'nu' : 345.8, # ghz
              'imres' : 0.005, # arcsec/pixel
              'PA' : 128, # degrees
              'distance' : 100} # parsecs

mod  = DD.Disk(inc=88.5, 
                  sh_params=[0.025], 
                  radial_params=[2.2], 
                  disk_edges=[22,42], 
                  obs=obs_params)

mod.square()
mod.rotate()
mod.save('example.fits')

tic = time.time()

print("TIME = ", (tic - toc))
