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
               radial_params=[2.2], 
               disk_edges=[22,42], # au 
               sh_func='linear', # options: constant
               sh_params=[0.025],  
               vert_func='gaussian', 
               obs=obs_params)

mod.square()
mod.rotate()
mod.save('example.fits')

vis.sample(mod)

tic = time.time()

print("TIME = ", (tic - toc))
