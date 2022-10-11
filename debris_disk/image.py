import numpy as np
from scipy import ndimage
from astropy.io import fits
from debris_disk import profiles

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
    rotate():
        Rotates self.val by PA in observation dictionary
    save():
        Saves self.val to output file
    primary_corr():
        Multiplies self.val by observation primary beam
    """

    def __init__(self, val, imres=None, axes=None):
        self.val = val
        self.nx = np.shape(val)[1]
        self.ny = np.shape(val)[0]

        if imres:
            self.imres = imres

        if axes:
            self.x = axes[0]
            self.y = axes[1]
        else:
            self._axes()

    def _axes(self):
        start = int(self.nx/2)
        self.x = self.imres*(np.linspace(-start, start-1, self.nx))
        
        start = int(self.ny/2)
        self.y = self.imres*(np.linspace(-start, start-1, self.ny))

    def square(self):
        if self.nx == self.ny:
            return
        assert self.nx > self.ny
        start = int((self.nx-self.ny)/2)+1
        end = int((self.nx+self.ny)/2)+1
        im = np.zeros((self.nx, self.nx))
        im[start:end] = self.val
        self.val = im
        
        self.ny = self.nx
        self.y = self.x

    def rotate(self, obs):
        self.val = ndimage.rotate(self.val, 90-obs.PA, reshape=False)

    def save(self, obs, outfile):
        hdu = fits.PrimaryHDU(self.val, obs.header(self.nx))
        hdu.writeto(outfile, overwrite=True, output_verify='fix')

    def beam_corr(self, obs):
        xx, yy = np.meshgrid(self.x, self.y)
        dst = np.sqrt(xx**2+yy**2)
        sigma = 1.13 * obs.lamb/obs.D / (2 * np.sqrt(2 * np.log(2)))

        norm = profiles.gaussian.norm(dst, sigma)**2
        self.beam = profiles.gaussian.val(dst, sigma) 
        self.val *= self.beam
