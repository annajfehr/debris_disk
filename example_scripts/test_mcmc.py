import sys
sys.path.append('/arc/projects/ARKS/vertical_structure/parametric_modeling/')
import debris_disk as DD
import time
import numpy as np

toc = time.time()

vis = DD.UVDataset('../TYC9340/12m.txt', mode='mcmc')
freqs = DD.constants.c / vis.chans

obs_params = {'nu' : freqs, # ghz
              'imres' : 0.03, # arcsec/pixel
              'distance' : 36.6} # parsecs

rad_params = {'alpha' : 1.1,
              'Rin' : 50,
              'Rout' : 200}

rad_scale = {'alpha' : 1.,
              'Rin' : 20.,
              'Rout' : 50.}

rad_ranges = {'alpha' : (-5.,5.),
              'Rin' : (10., 100.),
              'Rout' : (100., 400)}

chain = DD.MCMC(uvdata='../TYC9340/12m.txt',
                obs_params=obs_params,
                fixed_args={'L_star' : 14.93,
                            'radial_func' : 'powerlaw',
                            'vert_func' : 'gaussian',
                            'vert_params' : {'Rc' : 1,
                                             'psi' : 1.}},
                p0={'radial_params' : rad_params,
                    'vert_params' : {'Hc' : 0.025},
                    'sigma_crit' : 2e-17,
                    'PA' : -72,
                    'inc' : 79},
                pranges={'radial_params': rad_ranges,
                         'vert_params' : {'Hc' : (1e-8, 1)},
                         'sigma_crit' : (1e-50, 1e-10),
                         'PA' : (-90, 90),
                         'inc' : (1., 90.)},
                pscale={'radial_params' : rad_scale,
                        'vert_params' : {'Hc' : (0.05)},
                        'sigma_crit' : 1e-17,
                        'PA' : 20,
                        'inc' : 20})

chain.run(15, 3, parallel=True)
