'''
Disk object. Initializes from radial/vertical/scale height 
functions or from profiles, and produces 2D image
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
from scipy.integrate import cumtrapz,trapz
import debris_disk.constants as const
from debris_disk.image import Image
from debris_disk.observation import Observation
from debris_disk import profiles

class Disk:
    """Class for circumstellar disk structure."""
    def __init__(self,
                 Lstar=1.,
                 Mdust=1e-7,
                 inc=0,
                 radial_func='powerlaw',
                 radial_params=[1.],
                 disk_edges=None,
                 sh_func='linear',
                 sh_params=[0.1],
                 vert_func='gaussian',
                 vert_params=None,
                 obs=None):
        # Set Conditions
        if obs:
            if type(obs) == dict:
                obs = Observation(**obs)
            self.obs = obs
            self.imres = obs.imres * obs.distance * const.AU
        else:
            self.imres = 0.5 * const.AU

        self.Lstar = Lstar
        self.Mdust = Mdust*const.Msun 
        self.inc = inc * np.pi / 180.

        self.disk_edges = disk_edges

        self.radial_func = radial_func
        self.radial_params = radial_params

        self.vert_func = vert_func

        self.sh_func = sh_func
        self.sh_params = sh_params

        self.surface_density()  # create 2d disk density structure
        self.incline() # produce 3d sky plane density
        
        if obs:
            self.integrate(obs) # integrate to find image intensities

    def radial_bounds(self):
        '''Find the inner and outer radius of the disk'''
        if self.disk_edges:
            self.disk_edges = [edge * const.AU for edge in self.disk_edges]
            self.rbounds = [(edge - 100) for edge in self.disk_edges]
            return
        if self.radial_func == 'gaussian':
            self.rbounds = profiles.gaussian.limits(*self.radial_params)
            return
        raise Exception("No bounds given and cannot find bounds from functional form")

    def surface_density(self):
        '''Calculate the disk density and temperature structure given the specified parameters'''
        self.radial_bounds()
        self.nr = int(10 * (self.rbounds[1] - self.rbounds[0]) / self.imres)
        if self.radial_func == 'powerlaw':
            self.r   = np.logspace(np.log10(self.rbounds[0]),
                                   np.log10(self.rbounds[1]), 
                                   self.nr)
        else:
            self.r = np.linspace(self.rbounds[0], self.rbounds[1], self.nr)
        
        assert self.find_H() # Calculates scale height (i.e. standard deviation)
                             # as a function of r
        
        self.find_zmax()
        self.nz = int(10 * self.zmax / self.imres)
        self.z   = np.linspace(0, 
                                self.zmax, 
                                self.nz)
        self.zz, self.rr = np.meshgrid(self.z, self.r)
        
        assert self.find_sigma() # Populates sigmaD, surface density ignoring
                                 # vertical structure using radial_func
        assert self.find_vert() # Populates vertD, vertical density structure
                                # using self.H and vert_func
        assert self.find_T() # Calculate 2d temperature array

        self.rho2D = self.sigmaD * self.vertD # Multiply radial and vertical
                                             # density structures
        np.nan_to_num(self.rho2D, nan=1e-60) # Check for nans

    def find_H(self):
        if self.sh_func=='linear':
            self.H = profiles.linear.val(self.r, *self.sh_params)
            self.Hnorm = profiles.linear.norm(*self.sh_params, *self.rbounds)
            return True

    def find_sigma(self):
        if self.radial_func == 'powerlaw':
            val = profiles.powerlaw.val(self.rr, *self.radial_params)
            norm = profiles.powerlaw.norm(*self.radial_params, *self.disk_edges)
        
        if self.radial_func == 'gaussian':
            val = profiles.gaussian.val(self.rr, *self.radial_params)
            norm = profiles.gaussian.norm(*self.radial_params)
        
        self.sigmaD = (self.Mdust/(2*np.pi*norm)) * val
        
        if self.disk_edges:
            empty_indices = (self.rr < self.disk_edges[0]) | (self.rr >
                    self.disk_edges[1])
            self.sigmaD[empty_indices] = 0
        return True

    def find_vert(self):
        if self.vert_func =='gaussian':
            H2D = np.outer(self.H, np.ones(self.nz))
            self.vertD = profiles.gaussian.val(self.zz, H2D)/(self.Hnorm*np.sqrt(np.pi))
            return True
    
    def find_T(self):
        self.T = (self.Lstar * const.Lsun / (16. * np.pi * self.rr**2 * const.sigmaB))**0.25
        return True
    
    def find_zmax(self):
        max_H = np.max(self.H)
        test_zs = np.linspace(0, 100 * const.AU, 1000)
        if self.vert_func == 'gaussian':
            max_vert = profiles.gaussian.val(test_zs, max_H *np.ones(1000))
        density_limit = np.max(max_vert) * 0.01
        self.zmax = test_zs[np.argmin(abs(max_vert - density_limit))]

    def incline(self):
        cosinc = np.cos(self.inc)
        sininc = np.sin(self.inc) 

        Slim = ((self.rbounds[1] * sininc) + (self.zmax * cosinc))
        Xlim = self.rbounds[1]
        Ylim = ((self.rbounds[1] * cosinc) + (self.zmax * sininc))

        self.nX = int(2 * Xlim / self.imres)
        self.nY = int(2 * Ylim / self.imres)

        if self.nX % 2 != 0:
            self.nX += 1 # Makes sure that output image is compatible with
                         # Galario -- nY will match nX by then so don't worry
                         # about it

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
        self.T=ndimage.map_coordinates(self.T,[[xind],[yind]],order=1).reshape(self.nY,self.nX, nS) 
        self.rhoD=ndimage.map_coordinates(self.rho2D,[[xind],[yind]],order=1).reshape(self.nY, self.nX, nS)  

    def integrate(self, obs):
        kap = 10 * (obs.nu/1e12)**const.beta
        
        BBF1 = 2.*const.h/(const.c**2)             # - prefactor for BB function
        BBF2 = const.h/const.kB                    # - exponent prefactor for BB function


        Knu_dust = kap*self.rhoD      # - dust absorbing coefficient
        Snu = BBF1*obs.nu**3/(np.exp((BBF2*obs.nu)/self.T)-1.) # - source function
        
        tau = cumtrapz(Knu_dust, self.S, axis=2, initial=0.)
        arg = Knu_dust*Snu*np.exp(-tau)

        self.im = Image(trapz(arg, self.S, axis=2), obs.imres)

    def square(self):
        self.im.square()

    def rotate(self):
        self.im.rotate(self.obs)

    def save(self, outfile='model.fits'):
        self.im.save(self.obs, outfile)

    def image(self):
        return self.im
