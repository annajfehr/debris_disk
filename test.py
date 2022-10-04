import model
import time

toc = time.time()

niter = 1
for _ in range(niter):
    obs = model.Observation(imres=0.005, PA = 128)
    im  = model.Disk(inc=88.5, 
                          sh_params=[0.025], 
                          radial_params=[2.2], 
                          disk_edges=[22,42], 
                          obs=obs).im
    im.square()
    im.rotate(obs)
    im.save(obs)

tic = time.time()

print("TIME = ", (tic - toc)/niter)
