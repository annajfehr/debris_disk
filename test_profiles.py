import debris_disk as DD
import time
import numpy as np
import debris_disk.constants as const
import debris_disk.profiles as prof
from matplotlib import pyplot as plt

data_fps_list = ['/arc/projects/ARKS/data/products/HD170773/visibilities/HD170773.12m.continuum.fav.tav.corrected.txt']
vis = DD.UVDataset(data_fps_list)

obs_params = DD.Observation(vis=vis,                              
                            json_file='/arc/projects/ARKS/parametric_modeling/REASONS.json',
                            sys_name='HD170773')

#radfunc = 'gauss_dpl'
#rad_params = {'sigma': 50.0, 'R_gauss': 0.0, 'rc': 100.0, 'alpha_in': 5.0, 'alpha_out': -3.0, 'gamma': 2}

#radfunc = 'double_gaussian'
#rad_params = {'R1': 50.0, 'R2': 100.0, 'sigma1': 20.0, 'sigma2': 30.0, 'C1': 1.0, 'C2': 0.5}
#rad_params = {'R1': 50.0, 'R2': 100.0, 'sigma1': 20.0, 'sigma2': 30.0}

#radfunc = 'triple_gaussian'
#rad_params = {'R1': 50.0, 'R2': 100.0, 'R3': 120, 'sigma1': 20.0, 'sigma2': 30.0, 'sigma3': 5.0, 'C1': 1.0, 'C2': 0.5, 'C3': 0.8}
#rad_params = {'R1': 50.0, 'R2': 100.0, 'R3': 120, 'sigma1': 20.0, 'sigma2': 30.0, 'sigma3': 5.0}

radfunc = 'dpl_2gaussgaps'
#rad_params = {'rc': 70, 'alpha_in': 5.0, 'alpha_out': -8.0, 'gamma': 2.0, 'R1': 55.0, 'R2': 85.0, 'sigma1': 3.0, 'sigma2': 2.0, 'C1': 0.15, 'C2': 0.2}
rad_params = {'rc': 70, 'alpha_in': 5.0, 'alpha_out': -8.0, 'gamma': 2.0, 'R1': 55.0, 'R2': 85.0, 'sigma1': 3.0, 'sigma2': 2.0}


vert_params = {'Hc' : .03,
               'Rc' : 1,
               'psi' : 1.}

mod  = DD.Disk(L_star=1, # L_sun
               inc=0, # Degrees
               sigma_crit = 1e-59,
               radial_func=radfunc,
               radial_params=rad_params,
               vert_func='gaussian',
               vert_params=vert_params,
               obs=obs_params,
               rmax=7)

fig, ax = plt.subplots(2)
ax[0].plot(mod.r/const.AU, np.sum(mod.rho2d, axis=0))
ax[0].set_xlabel('r [au]')
ax[0].set_ylabel('density [arbitrary units]')
ax[1].imshow(mod.ims[0].val)
plt.tight_layout()
plt.savefig(radfunc+'_test.png')