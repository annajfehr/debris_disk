import numpy as np
from scipy import ndimage
from astropy.io import fits
from debris_disk import profiles
from debris_disk import constants as const

class Image:
    """
    A class to represent an image.

    ...

    Attributes
    ----------
    val : array of floats
        Array containing image brightness
    nx : int
        Length of val x axis
    ny : int
        Length of val y axis
    imres : float, optional
        Size of each pixel in val

    Methods
    -------
    square():
        Converts self.val to a square with sides nx
    shift(PA=0., dra=0., ddec=0):
        Rotates self.val by PA in observation dictionary
    save():
        Saves self.val to output file
    primary_corr():
        Multiplies self.val by observation primary beam
    """

    def __init__(self, val, imres=None, axes=None, modres=None):
        print("in image.py, val")
        print(np.mean(val)) # erg / (cm^2 s Hz) ?
        self.val = val * 1e23 # Jy/ster
        print("in image.py, val2 (possibly Jy/ster?)")
        # per pixel?
        print(np.mean(self.val))
        
        self.nx = np.shape(val)[1]
        self.ny = np.shape(val)[0]

        if modres:
            print("modres is being passed in")
            print(modres)
            print(np.mean(self.val))
            #self.val *= modres*modres # why would we do this? would be some weird unit
            #print(modres*modres)
            #self.val /= modres*modres # this would convert to Jy/cm^2 (I think?)
            #print("now in Jy/cm^2????")
            #self.val /= imres*imres
            # I DON'T KNOW WHAT THE UNITS OF THE IMAGE SHOULD BE
            # I think they are already in Jy/pixel so what if I just...comment this out?
            self.val *= ((imres**2) / (const.rad)**2) # if started with Jy/ster, now in Jy/pixel?
            print("did i fix it? let's see idk")
            print(np.mean(self.val))
        if imres:
            self.imres = imres
            print("imres is being passed in")
            print(self.imres)

        if axes:
            self.x = axes[0] # array of x pixel values in arcsec
            self.y = axes[1] # array of y pixel values in arcsec
        else:
            self._axes()
        print("self x and y")
        print(self.x)
        print(self.y)
        print(np.shape(self.x))
        print(np.shape(self.y))
        print("nx and ny")
        print(self.nx)
        print(self.ny)

    def _axes(self):
        start = int(self.nx/2)
        self.x = self.imres*(np.linspace(-start, start-1, self.nx))
        print("start: ", start)
        
        start = int(self.ny/2)
        self.y = self.imres*(np.linspace(-start, start-1, self.ny))
        print("start: ", start)

    def square(self):
        if self.nx == self.ny:
            return
        
        if self.nx > self.ny:
            start = int((self.nx-self.ny)/2)+1
            end = int((self.nx+self.ny)/2)+1
            im = np.zeros((self.nx, self.nx))
            im[start:end] = self.val
            self.val = im
            
            self.ny = self.nx
            self.y = self.x
        else:
            start = int((self.ny-self.nx)/2)+1
            end = int((self.nx+self.ny)/2)+1
            im = np.zeros((self.ny, self.ny))
            im[:,start:end] = self.val
            self.val = im
            
            self.nx = self.ny
            self.x = self.y

    def rotate(self, PA=0., dra=0., ddec=0.):
        self.val = ndimage.rotate(self.val, 90-PA, reshape=False)
        zeros = np.zeros(np.shape(self.val))
        self.val = np.maximum(self.val, zeros)

    def save(self, obs, outfile):
        hdu = fits.PrimaryHDU(self.val, obs.header(self.nx))
        hdu.writeto(outfile, overwrite=True, output_verify='fix')

    def beam_corr(self, nu, D):
        lamb = (const.c/100) / nu # wavelength in [m]
        xx, yy = np.meshgrid(self.x, self.y)
        dst = np.sqrt(xx**2+yy**2) # in arcseconds
        sigma = 206265 * 1.13 * lamb / D / (2 * np.sqrt(2 * np.log(2))) # in arcsec
        norm = profiles.gaussian.norm(dst, sigma)**2
        self.beam = profiles.gaussian.val(dst, sigma) 
        self.val *= self.beam

    def add_star(self, f_star):
        # add star at center of image
        self.val[int(self.ny/2), int(self.nx/2)] += f_star # in Jy
