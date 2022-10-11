import debris_disk.constants as const
from astropy.io import fits
import numpy as np


class Observation:
    def __init__(self,
                 nu = 345.8,
                 imres = 0.005,
                 distance = 100.,
                 PA = 0.,
                 vis_file=None):
        if vis_file:
            pass
        else:
            self.nu = nu * const.Ghz
            self.lamb = 3e8 / nu # wavelength [m]
            self.imres = imres
            
        self.distance = distance
        self.PA = PA
        self.D = 12. # antennae diameter [m]
    
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
