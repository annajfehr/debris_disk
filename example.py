import model
import time
import numpy as np

toc = time.time()


vis = model.UVData('../hd106906/mcmc_data/', mode='MCMC')
    
obs = model.Observation(imres=0.005, PA = 128)
im  = model.Disk(inc=88.5, 
                      sh_params=[0.025], 
                      radial_params=[2.2], 
                      disk_edges=[22,42], 
                      obs=obs).im
im.square()
im.rotate(obs)

#vis.sample(im.val, im.imres * np.pi / (180*3600))
print(vis.chi2(im.val, im.imres * np.pi / (180*3600)))
tic = time.time()

print("TIME = ", (tic - toc))
