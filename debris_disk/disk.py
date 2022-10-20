import numpy as np
from scipy import ndimage
from scipy.integrate import cumtrapz,trapezoid
import debris_disk.constants as const
from debris_disk.image import Image
from debris_disk.observation import Observation
from debris_disk import profiles

class Disk:
    """
    A class to represent a disk.

    ...

    Attributes
    ----------
    L_star : float
        Luminosity of the star [solar luminosities]
    sigma_crit : float
        Density of the disk at the critical radius [g/cm^2]
    inc : float, 
        Inclination of the disk relative to the viewer [degrees]
    radial_func : {'powerlaw', 'gaussian', 'double_powerlaw',
                   'triple_powerlaw'}
        Functional form of the radial profile
    radial_params : dict
        Arguments for radial profile
    gap : bool, optional
        Whether to include a Gaussian gap in the radial structure
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
    vert_func : {'gaussian', 'lorentzian'}
        Functional form of the vertical profile
    obs : dictionary of observation keywords, optional
        obs is required to produce an on-sky image, see
        debris_disk.observation.Observation
    rbounds : list of floats
        Inner and outer radius of disk extent [cm]
    zmax : float
        Maximum vertical extent of disk [cm]
    nr : int
        Number of pixels along r axis in initial grid
    nz : int
        Number of pixels along z axis in initial grid
    rho2d : array of floats
        nr x nz array of surface densities at rr, zz positions
    T2d : array of floats
        nr x nz array of temperatures at rr, zz positions
    nx : int
        Number of pixels along x axis of final image
    ny : int
        Number of pixels along y axis of final image
    nS : int
        Number of pixels along line of sight of 3d density array
    S : array of floats
        nx x ny x ns array of line of sight distance
    rho : array of floats
        nx x ny x nX array of density
    T : array of floats
        nx x ny x nS array of temperature
    im : Image object
        contains image of disk on sky brightness

    Methods
    -------
    square():
        Converts disk image to a square
    rotate():
        Rotates disk image using observation dictionary
    save():
        Saves disk image
    image():
        Returns disk image
    """
    def __init__(self,
                 L_star=1.,
                 sigma_crit = 1e-18, # g/cm^3
                 inc=0,
                 radial_func='powerlaw',
                 radial_params={'alpha' : 1., 
                                'Rin' : 10 * const.AU, 
                                'Rout' : 40 * const.AU},
                 gap=False,
                 gap_params=None,
                 vert_params={'Hc' : 1.,
                              'Rc' : 1.,
                              'psi' : 1.},
                 vert_func='gaussian',
                 obs=None):
        """
        Constructs disk object, including density arrays and on sky image, if
        given sufficient parameters.

        Parameters
        ----------
        L_star : float, optional
            Luminosity of the star [solar luminosities], default is 1.
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

        self.gap = gap
        self.gap_params = gap_params

        self.vert_func = vert_func

        self.vert_params = vert_params
        #self.scale_height = scale_height
        #self.aspect_ratio = aspect_ratio

        self.structure2d()
        self.incline() # Produce 3d sky plane density
        
        if obs:
            self._im(obs) # Integrate to find image intensities

    def structure2d(self):
        """
        Initialize density and temperature structures
        """

        self._rbounds() # Find radial extent
        
        # Set number of pixels in 2d radial array to 10 times the desired final
        # image resolution
        self.nr = int(10 * (self.rbounds[1] - self.rbounds[0]) / self.imres) 

        # Define radial sampling grid -- for 'powerlaw' profile use logspaced
        # grid, in all other cases use uniform sampling
#        if self.radial_func == 'powerlaw':
#            r = np.logspace(np.log10(self.rbounds[0]),
#                                   np.log10(self.rbounds[1]), 
#                                   self.nr)
#        else:
        r = np.linspace(self.rbounds[0], self.rbounds[1], self.nr)
        
        # Calculates scale height as a function of r
        H, Hnorm = self.H(r, **self.vert_params)

        self._zmax(H) # Find vertical extent
        self.nz = int(10 * self.zmax / self.imres) # 10x final image resolution
        z = np.linspace(0, self.zmax, self.nz)
        
        rr, zz = np.meshgrid(r, z)
        assert self._rho2d(rr, zz, H, Hnorm)  # create 2d disk density structure
        assert self._T2d(rr) # Calculate 2d temperature array

    def _rbounds(self):
        """Set self.rbounds, the inner and outer radii of the disk where the
        brightness is 0.001 * the peak
        
        Returns
        -------
        None
        """

        if self.radial_func == 'gaussian':
            self.rbounds = profiles.gaussian.limits(**self.radial_params)
        
        if self.radial_func == 'powerlaw':
            self.rbounds = profiles.powerlaw.limits(**self.radial_params)

        if self.radial_func == 'double_powerlaw':
            self.rbounds = profiles.double_powerlaw.limits(**self.radial_params)

        if self.radial_func == 'triple_powerlaw':
            self.rbounds = profiles.triple_powerlaw.limits(**self.radial_params)
        
        self.rbounds[1] = min(self.rbounds[1], 250 * const.AU)
        assert (self.rbounds[0]>=0) and (self.rbounds[1]>self.rbounds[0]), "Cannot find bounds from functional form"
    
    def H(self, r, Hc, Rc, psi):
        """
        Determine scale height values

        Parameters
        ----------
        r : array of floats
            a 1d array of distance from disk center, with len(r) = self.nr

        Returns
        -------
        (H, Hnorm)
        
        H : array of floats
            a 1d array of the scale height at the positions in r
        Hnorm : float    
            integral of H
        """
        H = Hc * (r / Rc)**psi
        self.Harr = H
        self.r = r
        Hnorm = (Hc / (psi+1)) * ((self.rbounds[1] / Rc)**(psi+1) - \
                (self.rbounds[0] / Rc)**(psi+1))
        return H, Hnorm
    
    def _zmax(self, H):
        """
        Set self.zmax, the height where the disk brightness is 0.001 * the peak

        Parameters
        ----------
        H : array of floats
            scale height at all radii values

        Returns
        -------
        None
        """

        if self.vert_func == 'gaussian':
            _, self.zmax = profiles.gaussian.limits(H[-1])
        
        if self.vert_func == 'lorentzian':
            _, self.zmax = profiles.lorentzian.limits(H[-1])
    
    def _rho2d(self, rr, zz, H, Hnorm):
        """
        Initialize self.rho2d, a nr x nz matrix where self.rho2d[i, j] is the
        surface density of the disk (g/cm^2) at the location described by
        self.zz[i], self.zz[j]

        Parameters
        ---------
        rr : array of floats
            nr x nz array containing distance from disk center
        zz : array of floats
            nr x nz array containing distance from disk midplane
        H : array of floats
            len nr array containing disk scale height
        Hnorm : float
            integral of H

        Returns
        -------
        True upon success
        """
        
        # Radial x vertical density structure
        self.rho2d = self.sigma(rr) * self.vert(zz, H, Hnorm) 
        np.nan_to_num(self.rho2d, nan=1e-60) # Check for nans
        return True

    def sigma(self, rr):
        """
        Produce radial density structure

        Parameters
        ----------
        rr : array of floats
            an nr x nz array of distances from the center of the disk

        Returns
        -------
        an nr x nz array where arr[i, j] is the value of the radial profile at
        self.r[i]
        """

        if self.radial_func == 'powerlaw':
            val = profiles.powerlaw.val(rr, **self.radial_params)

        if self.radial_func == 'double_powerlaw':
            val = profiles.double_powerlaw.val(rr, **self.radial_params)
        
        if self.radial_func == 'triple_powerlaw':
            val = profiles.triple_powerlaw.val(rr, **self.radial_params)

        if self.radial_func == 'gaussian':
            val = profiles.gaussian.val(rr, **self.radial_params)

        if self.gap:
            gap = 1-(self.gap_params.pop('depth') * \
                    profiles.gaussian.val(rr, **self.gap_params))
            val*=gap
            self.g = gap

        return self.sigma_crit * val

    def vert(self, zz, H, Hnorm):
        """
        Produce vertical density structure 

        Parameters
        ----------
        zz : array of floats
            an nr x nz array of distances from the disk midplane
        H : array of floats
            len nr array containing disk scale height
        Hnorm : float
            integral of H

        Returns
        -------
        an nr x nz array where arr[i, j] is the normalized value of the vertical 
        profile at (self.r[i], self.z[j])
        """

        H2d = np.outer(np.ones(self.nz), H/(2*np.sqrt(2*np.log(2))))
        
        if self.vert_func =='gaussian':
            return profiles.gaussian.val(zz, H2d)/(Hnorm*np.sqrt(2*np.pi))
    
        if self.vert_func =='lorentzian':
            return 2*np.pi*profiles.gaussian.val(zz, H2d)/(Hnorm)
    
    def _T2d(self, rr):
        """
        Populate self.T2d, an nr x nz array where self.T2d[i, j] is the
        temperature at (self.r[i], self.z[j])

        Parameters
        ----------
        rr : array of floats
            an nr x nz array of distances from the center of the disk
        
        Returns
        -------
        True upon success
        """

        self.T2d = (self.L_star * const.Lsun / (16. * np.pi * rr**2 * const.sigmaB))**0.25
        return True

    def incline(self):
        """
        Populate 3d density and temperature grids by inclining the 2d grids.
        The resulting grids (self.rho and self.T) are in cartesian coordinates
        relative to the sky plane

        Returns
        -------
        None
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
        
        #re-define disk midplane coordinates to be in line with radiative transfer grid
        tz = yy*sininc+self.S*cosinc # z locations of points
        ty = yy*cosinc-self.S*sininc # y locations of points if disk was face on
        tr = np.sqrt(xx**2+ty**2)    # r locations of points

#        if self.radial_func == 'powerlaw':
#            slope = (self.nr-1)/np.log10(self.rbounds[1]/self.rbounds[0])
#            rind = slope * np.log10(tr/self.rbounds[0])
#        else:
        slope = (self.nr-1)/(self.rbounds[1]-self.rbounds[0])
        rind = slope * (tr-self.rbounds[0])

        slope = (self.nz-1)/(self.zmax)
        zind = slope * np.abs(tz)
        
        #interpolate onto coordinates xind,yind 
        self.T=ndimage.map_coordinates(self.T2d,[[zind],[rind]],order=1).reshape(self.nY,self.nX, nS) 
        self.rho=ndimage.map_coordinates(self.rho2d,[[zind],[rind]],order=1).reshape(self.nY, self.nX, nS)  

    def _im(self, obs):
        """
        Initialize an image object of the on-sky disk brightness using
        Rosenfeld+13 eq. 1

        Returns
        -------
        None
        """

        kap = 10 * (obs.nu/1e12)**const.beta
        
        BBF1 = 2.*const.h/(const.c**2)  # - prefactor for BB function
        BBF2 = const.h/const.kB         # - exponent prefactor for BB function


        Knu_dust = kap*self.rho      # - dust absorbing coefficient
        Snu = BBF1*obs.nu**3/(np.exp((BBF2*obs.nu)/self.T)-1.) # - source function
        
        tau = cumtrapz(Knu_dust, self.S, axis=2, initial=0.)
        arg = Knu_dust*Snu*np.exp(-tau)
        self.im = Image(trapezoid(arg, self.S, axis=2), obs.imres)

    def square(self):
        """
        Alias for debris_disk.image.Image.square() 
        Expands self.im to produce a square image

        Returns
        -------
        None
        """

        self.im.square()

    def shift(self, PA=0.):
        """
        Alias for debris_disk.image.Image.rotate()
        Rotates self.im by the position angle
        
        Returns
        -------
        None
        """

        self.im.rotate(PA)

    def save(self, outfile='model.fits'):
        """
        Alias for debris_disk.image.Image.save()
        Saves self.im at outfile

        Returns
        -------
        None
        """

        self.im.save(self.obs, outfile)

    def image(self):
        """
        Returns self.im

        Returns
        -------
        Image object
        """
    
        return self.im

    def beam_corr(self):
        """
        Alias for debris_disk.image.Image.beam_corr()
        Applies primary beam correction

        Returns
        -------
        None
        """

        self.im.beam_corr(self.obs)
