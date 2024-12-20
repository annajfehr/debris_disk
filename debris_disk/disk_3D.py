import numpy as np
from scipy import ndimage
from scipy.integrate import cumtrapz,trapezoid
import debris_disk.constants as const
from debris_disk.image import Image
from debris_disk.observation import Observation
from debris_disk import profiles
# from memory_profiler import profile

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
                 inc=0, # degrees
                 F_star=0, #microjy
                 radial_func='powerlaw',
                 radial_params={'alpha' : 1., 
                                'Rin' : 10, 
                                'Rout' : 40},
                 rmax=None, # arcsec
                 ecc=0.,
                 aop=0.,
                 gap=False,
                 gap_params=None,
                 vert_params={'Hc' : 1.,
                              'Rc' : 1.,
                              'psi' : 0.},
                 vert_func='gaussian',
                 rbounds=None,
                 obs=None,
                 calc_structure=True,
                 calc_image=True,
                 memory=32):
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
        inc : float, optional
            Inclination of the disk relative to the viewer [degrees], default is 0. 
        F_star : float
            Emission from the star [microjy], default is 0
        radial_func : {'powerlaw', 'powerlaw_errf', 'double_powerlaw',
                       'triple_powerlaw'}, optional
            Functional form of the radial profile, default is 'powerlaw'
        radial_params : dictionary of float
            Arguments for radial profile, must match radial_func. All distances
            should be in au. Default is {'alpha' : 1, 'Rin' : 10, 'Rout' : 40}
        rmax : float or none
            Maximum radius to calculate the density grid, in arcsec
        ecc : float
            Eccentricity of the disk. Float between zero and one
        aop : float
            Argument of periastron, in radians. Measured counterclockwise from *West*.
        gap : bool, optional
            Whether to include a Gaussian gap in the radial structure, default
            is False
        gap_params : list of float, optional
            Required if gap. default is None
            gap_params['r'] : Radius of the gap's deepest point, au
            gap_params['width'] : FWHM max of the gap, au
            gap_params['depth'] : Fractional depth of the gap, [0, 1]
        vert_params : list of float, optional
            Parameters for vertical structure from Fehr2023 Eq. 4.2
            Default is {'Hc' : 1, 'Rc' : 1, 'psi' : 1}
        vert_func : {'gaussian', 'lorentzian'}, optional
            Functional form of the vertical profile, default is gaussian
        obs : dictionary of observation keywords, optional
            {'nu' : Central frequency of observation [ghz]
             'imres' : Resolution of final model image [arcsec/pixel]
             'distance' : Distance to object [parsecs]}
        calc_structure : boolean
            controls whether or not to calculate the disk structure upon
            initialization. Default is True
        calc_image : boolean
            controls whether to calculate synthetic image upon initialization.
            Default is True.
        """
        self.max_mem = .75 * memory/16 * 1e9

        if obs:
            if type(obs) == dict:
                obs = Observation(**obs)
            self.obs = obs
            self.modres = obs.imres * obs.distance * const.AU
            # self.max_r = obs.distance * 8 * const.AU
            self.max_r = obs.distance * obs.beam_fwhm * 3600 * 180 / np.pi * const.AU
        else:
            self.modres = 0.5 * const.AU
            self.max_r = 50 * 8 * const.AU
        if rmax:
            self.max_r = obs.distance * rmax * const.AU
        if 'Rin' in radial_params:
            if radial_params['Rin'] > radial_params['Rout']:
                self.mod = False
                return

        self.ecc=ecc
        self.aop=aop

        self.L_star = L_star
        self.F_star = F_star
        self.sigma_crit = sigma_crit
        
        if inc < 90:
            self.inc = inc * np.pi / 180.
        else:
            self.inc = (180 - inc) * np.pi/180

        self.radial_func = radial_func
        self.radial_params = self.rp_convert(radial_params.copy())
        
        if rbounds:
            self.rbounds=rbounds.copy()
        else:
            self.rbounds = rbounds

        self.gap = gap
        self.gap_params = gap_params

        self.vert_func = vert_func

        self.vert_params = self.vp_convert(vert_params.copy())

        if calc_structure:
            if not self.structure2d():
                return
            if not self.incline(): # Produce 3d sky plane density
                return
        else:
            return

        self.mod = True
        if obs and calc_image:
            self._im(obs) # Integrate to find image intensities

    def rp_convert(self, radial_params):
        def convert_func(func, params):
            if func == 'gaussian':
                return profiles.gaussian.conversion(params, const.AU)
            if func == 'powerlaw':
                return profiles.powerlaw.conversion(params, const.AU)
            if func == 'powerlaw_errf':
                return profiles.powerlaw_errf.conversion(params, const.AU)
            if func == 'double_powerlaw':
                return profiles.double_powerlaw.conversion(params, const.AU)
            if func == 'triple_powerlaw':
                return profiles.triple_powerlaw.conversion(params, const.AU)
            if func == 'single_erf':
                return profiles.single_erf.conversion(params, const.AU)
            if func == 'asymmetric_gaussian':
                return profiles.asymmetric_gaussian.conversion(params, const.AU)

        if type(self.radial_func) is list:
            for i, func in enumerate(self.radial_func):
                radial_params[i] = convert_func(func, radial_params[i].copy())
            return radial_params
        else:
            return convert_func(self.radial_func, radial_params)

    def vp_convert(self, vert_params):
        vert_params['Hc'] *= const.AU
        vert_params['Rc'] *= const.AU
        return vert_params

    def structure2d(self):
        """
        Initialize density and temperature structures
        """
        if self.rbounds:
            self.rbounds[0] *= const.AU
            self.rbounds[1] *= const.AU
        else:
            self._rbounds() # Find radial extent

        # calculate abounds from rbounds. For each orbit, periastron is a * (1-e), apoastron is a * (1+e)
        self.abounds = [self.rbounds[0]/(1-self.ecc), self.rbounds[1]/(1+self.ecc)]

        # Set number of pixels in 2d radial array to 5 times the desired final
        # image resolution
        
        self.na = int(5  * (self.abounds[1] - self.abounds[0]) / self.modres) 
        self.nphi = self.na # Should we calculate the number of pixels in phi a different way?

        # Define grid in semimajor axis
        self.a = np.linspace(self.abounds[0], self.abounds[1], self.na) # semimajor axis grid
        self.p = np.linspace(0., 2*np.pi, self.nphi) # azimuthal angel grid (from line of nodes)
 
        self.zmax=self.abounds[1]/10 # For now, let's assume that zmax < amax/10 (good for a thin disk, definitely look at this for a thick disk)
        self.nz = int(5 * self.zmax / self.modres) # 5x final image resolution
        self.z = np.linspace(0, self.zmax, self.nz)
        pp, aa, zz = np.meshgrid(self.p, self.a, self.z) # Make 3d grids
        self.aa = aa 
        rr = (aa*(1.-self.ecc*self.ecc)/(1.+self.ecc*np.cos(pp))) # Make r grid (important for temperature calculation)
        self.rr = rr
        vert_struct = self.H(**self.vert_params) # Calculate vertical structure (vert_struct is 3D)

        assert self._rho2d(pp, aa, zz, rr, vert_struct)  # create 2d disk density structure
        assert self._T2d(rr, zz) # Calculate 2d temperature array
        
        return True

    def _rbounds(self):
        """Set self.rbounds, the inner and outer radii of the disk where the
        brightness is 0.001 * the peak
        
        Returns
        -------
        None
        """
        self.rbounds =  [self.modres/2, self.max_r]
 
    def H(self, Hc, Rc, psi, gamma=2):
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
        H = Hc * (self.aa / Rc)**psi
        self.Harr = H
        Hnorm = (Hc / (psi+1)) * ((self.rbounds[1] / Rc)**(psi+1) - \
                (self.rbounds[0] / Rc)**(psi+1))
        return H, Hnorm, gamma
    
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

        if self.zmax < self.modres:
            self.zmax = self.modres/2

    def _rho2d(self, pp, aa, zz, rr, vert_struct):
        """
        Initialize self.rho2d, a np x na x nz matrix where self.rho2d[i, j, k] is the
        surface density of the disk (g/cm^2) at the location described by
        self.p[i], self.a[j], self.z[k]

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
        
        # Radial x vertical x azimuthal density structure
        self.rho2d = self.sigma(aa) * self.vert(zz, *vert_struct) * self.sigphi(aa, pp)
        
        np.nan_to_num(self.rho2d, nan=1e-60) # Check for nans
        return True

    def sigma(self, aa):
        """
        Produce radial density structure

        Parameters
        ----------
        aa : array of floats
            an np x na x nz array of semi-major axes from the center of the disk

        Returns
        -------
        an np x na x nz array where arr[i, j, k] is the value of the radial profile at
        self.a[j]
        """

        def profile_from_func(radial_func, params):
            if radial_func == 'powerlaw':
                return profiles.powerlaw.val(aa, **params)
            
            if radial_func == 'powerlaw_errf':
                return profiles.powerlaw_errf.val(aa, **params)

            if radial_func == 'double_powerlaw':
                return profiles.double_powerlaw.val(aa, **params)
            
            if radial_func == 'triple_powerlaw':
                return profiles.triple_powerlaw.val(aa, **params)

            if radial_func == 'gaussian':
                return profiles.gaussian.val(aa, **params)
            
            if radial_func == 'single_erf':
                return profiles.single_erf.val(aa, **params)
            
            if radial_func == 'asymmetric_gaussian':
                return profiles.asymmetric_gaussian.val(aa, **params)

        if type(self.radial_func) == list:
            val = np.zeros(np.shape(aa))
            for func, params in zip(self.radial_func, self.radial_params):
                try:
                    norm = params.pop('norm')
                except:
                    norm = 1
                val += norm * profile_from_func(func, params)
        else:
            val = profile_from_func(self.radial_func, self.radial_params)

        def make_gap(params):
            gap_params = profiles.gaussian.conversion(params, const.AU)
            gap = 1-(gap_params.pop('depth') * \
                    profiles.gaussian.val(aa, **gap_params))
            return gap

        if self.gap:
            if type(self.gap_params) is list:
                gap = np.zeros(np.shape(val))
                for params in self.gap_params:
                    gap += make_gap(params.copy())
                gap -= (len(self.gap_params) - 1)
            else:
                gap = make_gap(self.gap_params.copy())
            val*=gap
            self.g = gap

        self.sig= self.sigma_crit * val
        return self.sigma_crit * val

    def vert(self, zz, H, Hnorm, gamma=2):
        """
        Produce vertical density structure 

        Parameters
        ----------
        zz : array of floats
            an nr x nz array of distances from the disk midplane
        H : array of floats
     H       len nr array containing disk scale height
        Hnorm : float
            integral of H

        Returns
        -------
        an nr x nz array where arr[i, j] is the normalized value of the vertical 
        profile at (self.r[i], self.z[j])
        """

        H2d = H[:] / (2*np.sqrt(2*np.log(2)))
        self.H2d= H2d
        if self.vert_func =='gaussian':
            normalize = profiles.gaussian.norm(H2d)
            self.vert_arr = profiles.gaussian.val(zz, H2d)/normalize
 
        if self.vert_func =='lorentzian':
            self.vert_arr =  2*np.pi*profiles.gaussian.val(zz, H2d)/(Hnorm)

        #self.vert_arr /= np.outer(np.sum(self.vert_arr, axis=2), np.ones(self.nz)).reshape((self.na, self.na, self.nz))
        return self.vert_arr

    def sigphi(self, aa, pp):
        '''
        Produce azimuthal density structure

        Parameters
        ----------
        aa : array of floats
            an np x na x nz array giving the semimajor axis at each pixel
        pp : array of floats
            an np x na x nz array giving the azimuth at each pixel

        Returns 
        -------
        an np x na x nz array where arr[i. j , k] is the value of the azimuthal variation at
        (self.p[i], self.r[j]). In this implementation, eccentricity is constant with semi-major axis
        '''
        pp=(pp-self.aop)%(2.*np.pi) # subtract argument of periastron (rotates counter clockwise)

        # Equation from ???
        dsdth = (aa*(1-self.ecc*self.ecc)*np.sqrt(1+2*self.ecc*np.cos(pp)+self.ecc*self.ecc))/(1+self.ecc*np.cos(pp))**2
        return (np.sqrt(1.-self.ecc*self.ecc))/(aa*np.sqrt(1+2*self.ecc*np.cos(pp)+self.ecc*self.ecc))*dsdth
    
    def _T2d(self, rr, zz):
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
        dist = np.sqrt((rr**2) + (zz**2))
        self.dist = dist
        self.T2d = (self.L_star * const.Lsun / (16. * np.pi * (dist**2) * const.sigmaB))**0.25
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

        self.nX = int(2 * Xlim / self.modres)
        self.nY = int(2 * Ylim / self.modres)

        if self.nX % 2 != 0:
            self.nX += 1 # Require that the number of pixels along the long axis
                         # is even for compatability with galario
        if self.nY % 2 != 0:
            self.nY += 1

        nS = int(6 * Slim / self.modres)
        
        if self.nX * self. nY * nS * 8 * 5 > self.max_mem: # 8 bits/float, 5 copies of the array    
            print('memory too high')
            return False

        X = np.linspace(-Xlim, Xlim, self.nX)
        Y = np.linspace(-Ylim, Ylim, self.nY)
        S = np.linspace(-Slim, Slim, nS)

        self.Xlim= Xlim
        self.Ylim = Ylim
        self.Slim = Slim

        xx, yy, self.S = np.meshgrid(X, Y, S)

        # transform grid to disk coordinates
        tdiskZ = yy*sininc + self.S * cosinc # z value of each pixel on the X x Y x S grid
        tdiskY = yy*cosinc - sininc*S # y value (in disk coordinates) for each pixel on the X x Y x S grid
        tr = np.sqrt(xx**2+tdiskY**2) # r value (in disk coordinates) for each pixel on the X x Y x S grid
        tphi = np.arctan2(tdiskY,xx)%(2*np.pi) # phi value (in disk coordinates) for each pixel on the X x Y x S grid


        zind = np.interp(np.abs(tdiskZ).flatten(),self.z,range(self.nz)) # interpolate from X x Y x S grid to P x R x Z grid
        phiind = np.interp(tphi.flatten(),self.p,range(self.na))
        aind = np.interp((tr.flatten()*(1+self.ecc*np.cos(tphi.flatten()-self.aop)))/(1.-self.ecc**2),self.a,range(self.na),right=self.na)


        
        self.T=ndimage.map_coordinates(self.T2d,[[aind],[phiind],[zind]],order=5).reshape(self.nY,self.nX, nS) # Map from P x R x Z to X x Y x S
        self.T[self.T<=0] = np.min(self.T2d)
        self.rho=ndimage.map_coordinates(self.rho2d,[[aind],[phiind],[zind]],order=5).reshape(self.nY, self.nX, nS)  
        
        
        assert np.shape(self.S) == np.shape(self.T)
        assert np.shape(self.S) == np.shape(self.rho)
        return True

    def _im(self, obs):
        """
        Initialize an image object of the on-sky disk brightness using
        Rosenfeld+13a eq. 1

        Returns
        -------
        None
        """
        self.ims = []
    
        BBF1 = 2.*const.h/(const.c**2)  # - prefactor for BB function
        BBF2 = const.h/const.kB         # - exponent prefactor for BB function
        
        tau = cumtrapz(self.rho, self.S, axis=2, initial=0.)
        
        for n in obs.nu:
            kap = 10 * (n/1e12)**const.beta
            tau *= kap

            # Knu_dust = kap*self.rho      # - dust absorbing coefficient
            # Snu = BBF1*n**3/(np.exp((BBF2*n)/self.T)-1.) # - source function

            #arg = kap*self.rho*np.exp(-tau)*BBF1*n**3/(np.exp((BBF2*n)/self.T)-1.) # - source function
            self.ims.append(Image(trapezoid(kap*self.rho*np.exp(-tau)*BBF1*n**3/(np.exp((BBF2*n)/self.T)-1.), 
                                            self.S, 
                                            axis=2),
                                  obs.imres,
                                  modres=self.modres))
            self.ims[-1].add_star(self.F_star)

    def square(self):
        """
        Alias for debris_disk.image.Image.square() 
        Expands self.im to produce a square image

        Returns
        -------
        None
        """
        for im in self.ims:
            im.square()

    def shift(self, PA=0.):
        """
        Alias for debris_disk.image.Image.rotate()
        Rotates self.im by the position angle
        
        Returns
        -------
        None
        """
        
        for im in self.ims:
            im.rotate(PA)

    def save(self, outfile='model'):
        """
        Alias for debris_disk.image.Image.save()
        Saves self.im at outfile

        Returns
        -------
        None
        """
        for i, im in enumerate(self.ims):
            im.square()
            im.save(self.obs, outfile+str(i)+'.fits')

    def image(self):
        """
        Returns self.im

        Returns
        -------
        Image object
        """
    
        return self.ims

    def beam_corr(self, obs):
        """
        Alias for debris_disk.image.Image.beam_corr()
        Applies primary beam correction

        Returns
        -------
        None
        """
        for im, n in zip(ims, obs.nu):
            self.im.beam_corr(self.obs, n, 12)

    def show_rho2d(self, args={}):
        self.show_2d(self.rho2d, **args)

    def show_T2d(self, args=[]):
        self.show_2d(self.T2d, **args)

    def show_2d(self, arr, ax=None, fig=None, colorbar=False):
        from matplotlib import pyplot as plt
        if not ax:
            fig, ax = plt.subplots()
        cm = ax.imshow(arr, 
                       origin='lower',
                       interpolation='none',
                       extent=[self.rbounds[0]/const.AU, self.rbounds[1]/const.AU, 0, self.zmax/const.AU])
        ax.set_xlabel(r'$r$ [au]',fontsize=12)
        ax.set_ylabel(r'$z$ [au]',fontsize=12)
        if colorbar:
            cbar = fig.colorbar(cm, ax=ax, shrink=1.1, aspect=10)
            cbar.set_label(colorbar, fontsize=12)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(0.1, self.zmax/const.AU)
        ax.set_xlim(self.rbounds[0]/const.AU, self.rbounds[1]/const.AU-1)
        ax.set_aspect('auto')

    def show_3d(self, arr, axes=None, fig=None, colorbar=False):
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots(3)
        axes[0].imshow(np.mean(arr, axis=2),
                       interpolation='none',
                       extent=[-self.Xlim/const.AU, self.Xlim/const.AU, -self.Ylim/const.AU, self.Ylim/const.AU])
        axes[0].set_xlabel('X [au]', fontsize=12)
        axes[0].set_ylabel('Y [au]', fontsize=12)

        axes[1].imshow(np.mean(arr, axis=1),
                       interpolation='none',
                       extent=[-self.Slim/const.AU, self.Slim/const.AU, -self.Ylim/const.AU, self.Ylim/const.AU])
        axes[1].set_xlabel('S [au]', fontsize=12)
        axes[1].set_ylabel('Y [au]', fontsize=12)

        axes[2].imshow(np.rot90(np.mean(arr, axis=0)),
                       interpolation='none',
             extent=[-self.Xlim/const.AU, self.Xlim/const.AU, -self.Slim/const.AU, self.Slim/const.AU])
        axes[2].set_xlabel('X [au]', fontsize=12)
        axes[2].set_ylabel('S [au]', fontsize=12)

        plt.tight_layout()
