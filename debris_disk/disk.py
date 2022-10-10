"""
Disk object. Initializes from radial/vertical/scale height 
functions or from profiles, and produces 2D image
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
from scipy._im import cumtrapz,trapz
import debris_disk.constants as const
from debris_disk.image import Image
from debris_disk.observation import Observation
from debris_disk import profiles

class Disk:
    """Class for circumstellar disk structure."""
    def __init__(self,
                 L_star=1.,
                 sigma_crit = 1e-18,
                 inc=0,
                 radial_func='powerlaw',
                 radial_params=[1., 10 * const.AU, 40 * const.AU],
                 gap=False,
                 gap_params=None,
                 scale_height=None,
                 aspect_ratio=None,
                 vert_func='gaussian',
                 obs=None):
        """Initialize disk object, including density arrays and on sky image, if
        given sufficient parameters.

        Parameters
        ----------
        L_star : float, optional
            Luminosity of the star [Solar Luminosities], default is 1.
        sigma_crit : float, optional
            Density of the disk at the critical radius [g/cm^2], default is
            1e-18
        inc : float, 
            Inclination of the disk relative to the viewer [degrees], default is 0. 
        radial_func : {'powerlaw', 'gaussian', 'double_powerlaw',
                       'triple_powerlaw'}, optional
            Functional form of the radial profile, default is 'powerlaw'
        radial_params : list of float
            Arguments for radial profile, must match radial_func. All distances
            should be in cm. Default is [1., 10 * const.AU, 40 * const.AU].
        gap : bool, optional
            Whether to include a Gaussian gap in the radial structure, default
            is False
        gap_params : list of float, optional
            Required if gap. 
            gap_params[0] : Radius of the gap's deepest point, cm
            gap_params[1] : FWHM max of the gap, cm
            gap_params[2] : Fractional depth of the gap, [0, 1]
        scale_height : float, optional
            FWHM of the disk's vertical structure, scale_height or aspect_ratio
            is required
        aspect_ratio: float, optional
            Scale height = aspect_ratio * r. scale_height or aspect_ratio is
            required
        vert_func : {'gaussian', 'lorentzian'}, optional
            Functional form of the vertical profile, default is gaussian
        obs : dictionary of observation keywords, optional
            {'nu' : Central frequency of observation [ghz]
             'imres' : Resolution of final model image [arcsec/pixel]
             'PA' : Position angle of disk [degrees]
             'distance' : Distance to object [parsecs]}
            obs is required to produce an on-sky image, see
            debris_disk.observation.Observation
        
        """
        if obs:
            if type(obs) == dict:
                obs = Observation(**obs)
            self.obs = obs
            self.imres = obs.imres * obs.distance * const.AU
        else:
            self.imres = 0.5 * const.AU

        self.L_star = L_star
        self.sigma_crit = sigma_crit
        self.inc = inc * np.pi / 180.


        self.radial_func = radial_func
        self.radial_params = radial_params

        self.vert_func = vert_func

        self.scale_height = scale_height
        self.aspect_ratio = aspect_ratio

        assert self._rho2D()  # create 2d disk density structure
        assert self._T2D() # Calculate 2d temperature array
        
        self.incline() # Produce 3d sky plane density
        
        if obs:
            self._im(obs) # Integrate to find image intensities


    def _rho2D(self):
        """Initialize self.rho2D, a nr x nz matrix where self.rho2D[i, j] is the
        surface density of the disk (g/cm^2) at the location described by
        self.zz[i], self.zz[j]
        
        """
        self._rbounds() # Find radial extent
        
        # Set number of pixels in 2d radial array to 10 times the desired final
        # image resolution
        self.nr = int(10 * (self.rbounds[1] - self.rbounds[0]) / self.imres) 

        # Define radial sampling grid -- for 'powerlaw' profile use logspaced
        # grid, in all other cases use uniform sampling
        if self.radial_func == 'powerlaw':
            self.r   = np.logspace(np.log10(self.rbounds[0]),
                                   np.log10(self.rbounds[1]), 
                                   self.nr)
        else:
            self.r = np.linspace(self.rbounds[0], self.rbounds[1], self.nr)
        
        # Calculates scale height as a function of r
        assert self._H(), 'Missing scale height or aspect ratio' 
        
        self._zmax() # Find vertical extent
        self.nz = int(10 * self.zmax / self.imres) # 10x final image resolution
        self.z = np.linspace(0, self.zmax, self.nz)
        
        self.zz, self.rr = np.meshgrid(self.z, self.r)
        
        assert self._sigma() # Populates radial density structure
        assert self._vert() # Populates vertical density structure

        self.rho2D = self.sigma * self.vert # Multiply radial and vertical
                                             # density structures
        np.nan_to_num(self.rho2D, nan=1e-60) # Check for nans
        return True

    def _rbounds(self):
        """Set self.rbounds, the inner and outer radii of the disk where the
        brightness is 0.001 * the peak
        
        """
        if self.radial_func == 'gaussian':
            self.rbounds = profiles.gaussian.limits(*self.radial_params)
        
        if self.radial_func == 'powerlaw':
            self.rbounds = profiles.powerlaw_edges.limits(*self.radial_params)

        assert (self.rbounds[0]>0) and (self.rbounds[1]>self.rbounds[0]), "Cannot find bounds from functional form"
    
    def _zmax(self):
        """Set self.zmax, the height where the disk brightness is 0.001 * the
        peak

        """
        max_H = self.H[-1]
        
        if self.vert_func == 'gaussian':
            _, self.zmax = profiles.gaussian.limits(max_H)
    
    def _H(self):
        """Populate self.H, an array where self.H[i] is the scale height at
        self.r[i] and set self.Hnorm, the integral of the self.H
        
        """
        if self.aspect_ratio:
            if self.scale_height:
                warnings.warn('Given aspect ratio and scale height,\
                initializing vertical structure using fixed aspect ratio')
            self.H = profiles.linear.val(self.r, self.aspect_ratio)
            self.Hnorm = profiles.linear.norm(self.aspect_ratio, *self.rbounds)
            return True
        
        if self.scale_height:
            self.H = profiles.constant.val(self.r, self.scale_height)
            self.Hnorm = profiles.constant.norm(self.scale_height,
                                                *self.rbounds)
            return True
        
        return False

    def _sigma(self):
        """Populate self.sigma, an nr x nz array where self.sigma[i,:] is the
        value of the radial profile at self.r[i]
    
        """
        if self.radial_func == 'powerlaw':
            val = profiles.powerlaw_edges.val(self.rr, *self.radial_params)
        
        if self.radial_func == 'gaussian':
            val = profiles.gaussian.val(self.rr, *self.radial_params)

        self.sigma = self.sigma_crit * val
        return True

    def _vert(self):
        """Populate self.vert, an nr x nz array where self.vert[i, j] is the
        value of the vertical profile at (self.r[i], self.z[j])
        
        """
        if self.vert_func =='gaussian':
            H2D = np.outer(self.H, np.ones(self.nz))
            self.vert = profiles.gaussian.val(self.zz, H2D)/(self.Hnorm*np.sqrt(np.pi))
            return True
    
    def _T2D(self):
        """Populate self.T2D, an nr x nz array where self.T2D[i, j] is the
        temperature at (self.r[i], self.z[j])

        """
        self.T2D = (self.L_star * const.Lsun / (16. * np.pi * self.rr**2 * const.sigmaB))**0.25
        return True
    

    def incline(self):
        """Populate 3d density and temperature grids by inclining the 2d grids.
        The resulting grids (self.rho and self.T) are in cartesian coordinates
        relative to the sky plane

        """
        cosinc = np.cos(self.inc)
        sininc = np.sin(self.inc) 

        Slim = ((self.rbounds[1] * sininc) + (self.zmax * cosinc))
        Xlim = self.rbounds[1]
        Ylim = ((self.rbounds[1] * cosinc) + (self.zmax * sininc))

        self.nX = int(2 * Xlim / self.imres)
        self.nY = int(2 * Ylim / self.imres)

        if self.nX % 2 != 0:
            self.nX += 1 # Require that the number of pixels along the long axis
                         # is even for compatability with galario
        
        nS = int(2 * Slim / self.imres)
        
        X = np.linspace(-Xlim, Xlim, self.nX)
        Y = np.linspace(-Ylim, Ylim, self.nY)
        S = np.linspace(-Slim, Slim, nS)

        xx, yy, self.S = np.meshgrid(X, Y, S)

        self.X = xx[:,:,0]
        self.Y = yy[:,:,0]
        
        #re-define disk midplane coordinates to be in line with radiative transfer grid
        tz = yy*sininc+self.S*cosinc # z locations of points
        ty = yy*cosinc-self.S*sininc # y locations of points if disk was face on
        tr = np.sqrt(xx**2+ty**2)    # r locations of points

        if self.radial_func == 'powerlaw':
            slope = (self.nr-1)/np.log10(self.rbounds[1]/self.rbounds[0])
            xind = slope * np.log10(tr/self.rbounds[0])
        else:
            slope = (self.nr-1)/(self.rbounds[1]-self.rbounds[0])
            xind = slope * (tr-self.rbounds[0])

        slope = (self.nz-1)/(self.zmax)
        yind = slope * np.abs(tz)
        
        #interpolate onto coordinates xind,yind 
        self.T=ndimage.map_coordinates(self.T2D,[[xind],[yind]],order=1).reshape(self.nY,self.nX, nS) 
        self.rho=ndimage.map_coordinates(self.rho2D,[[xind],[yind]],order=1).reshape(self.nY, self.nX, nS)  

    def _im(self, obs):
        """
        Initialize an image object of the on-sky disk brightness using
        Rosenfeld+13 eq. 1

        """
        kap = 10 * (obs.nu/1e12)**const.beta
        
        BBF1 = 2.*const.h/(const.c**2)             # - prefactor for BB function
        BBF2 = const.h/const.kB                    # - exponent prefactor for BB function


        Knu_dust = kap*self.rho      # - dust absorbing coefficient
        Snu = BBF1*obs.nu**3/(np.exp((BBF2*obs.nu)/self.T)-1.) # - source function
        
        tau = cumtrapz(Knu_dust, self.S, axis=2, initial=0.)
        arg = Knu_dust*Snu*np.exp(-tau)

        self.im = Image(trapz(arg, self.S, axis=2), obs.imres)

    def square(self):
        """
        Alias for debris_disk.image.Image.square() 
        Expands self.im to produce a square image
        
        """
        self.im.square()

    def rotate(self):
        """
        Alias for debris_disk.image.Image.rotate()
        Rotates self.im by the position angle

        """
        self.im.rotate(self.obs)

    def save(self, outfile='model.fits'):
        """
        Alias for debris_disk.image.Image.save()
        Saves self.im at outfile

        """
        self.im.save(self.obs, outfile)

    def image(self):
        """
        Returns self.im

        """
        return self.im
