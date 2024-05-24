import json
import debris_disk.constants as const
from astropy.io import fits
import numpy as np


class Observation:
    def __init__(self,
                 nu = 345.8,
                 imres = 0.005,
                 distance = 100.,
                 vis=None,
                 json_file=None,
                 sys_name=None,
                 Lstar=None,
                 lamb=None,
                 D=12):
        if vis:
            self.nu = const.c / vis.chans
            print("visbility resolutions!")
            print(vis.resolution)
            self.imres = min([np.min(res*(180/np.pi) * 3600/2) for res in vis.resolution]) # in arcseconds
            print("new imres")
            print(self.imres)
        else:
            self.nu = nu 
            self.imres=imres
        if json_file and sys_name:
            f = open(json_file)
            sys_props = json.load(f)
            self.distance = sys_props[sys_name]['d']['median']
            self.Lstar = sys_props[sys_name]['Lstar']['median']
        else:
            self.distance = distance
        self.lamb = 3e8 / nu # wavelength [m]
        self.D = D # antennae diameter [m]
        self.beam_fwhm = np.min(1.13 * self.lamb/self.D)
    
    def header(self, nX):
        '''
        Produce Image fits header, assumes that NAXIS1 = NAXIS2 = nX
        '''
        hdr = fits.Header()
        cen = [nX/2.+.5, nX/2.+.5]   # - central pixel location

        hdr['SIMPLE']='T'
        hdr['BITPIX'] = 32
        hdr['NAXIS'] = 3
        hdr['NAXIS1'] = nX
        hdr['NAXIS2'] = nX
        hdr['CDELT1'] = -1.*self.imres/3600.
        hdr['CRPIX1'] = cen[0]
        hdr['CRVAL1'] = 0
        hdr['CTYPE1'] = 'RA---SIN'
        hdr['CDELT2'] = self.imres/3600.
        hdr['CRPIX2'] = cen[1]
        hdr['CRVAL2'] = 0
        hdr['CTYPE2'] = 'DEC--SIN'
        hdr['OBJECT'] = 'model'

        return hdr
