import sys
sys.path.append('/Volumes/disks/anna/arks/')
import debris_disk as DD
import time
import numpy as np

toc = time.time()

vis = DD.UVData('../49ceti_fits/', mode='mcmc')

obs_params = {'nu' : vis.datasets[0].freq0/1e9, # ghz
              'imres' : 0.03, # arcsec/pixel
              'distance' : 57.2} # parsecs

rad_params = {'alpha' : 1.1,
              'Rin' : 50 * DD.constants.AU,
              'Rout' : 200 * DD.constants.AU}

rad_scale = {'alpha' : 1.,
              'Rin' : 20. * DD.constants.AU,
              'Rout' : 50. * DD.constants.AU}

rad_ranges = {'alpha' : (-5.,5.),
              'Rin' : (10. * DD.constants.AU, 100. * DD.constants.AU),
              'Rout' : (100. * DD.constants.AU, 400 * DD.constants.AU)}

chain = DD.MCMC(uvdata='../49ceti_fits/',
                obs_params=obs_params,
                fixed_args={'L_star' : 14.93,
                            'radial_func' : 'powerlaw',
                            'vert_func' : 'gaussian',
                            'vert_params' : {'Rc' : 1 * DD.constants.AU,
                                             'psi' : 1.}},
                p0={'radial_params' : rad_params,
                    'vert_params' : {'Hc' : 0.025 * DD.constants.AU},
                    'sigma_crit' : 2e-17,
                    'PA' : -72,
                    'inc' : 79},
                pranges={'radial_params': rad_ranges,
                         'vert_params' : {'Hc' : (1e-8, 1 * DD.constants.AU)},
                         'sigma_crit' : (1e-50, 1e-10),
                         'PA' : (-90, 90),
                         'inc' : (1., 90.)},
                pscale={'radial_params' : rad_scale,
                        'vert_params' : {'Hc' : (0.05 * DD.constants.AU)},
                        'sigma_crit' : 1e-17,
                        'PA' : 20,
                        'inc' : 20})

chain.run(50, 2500, parallel=False)
