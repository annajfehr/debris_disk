import numpy as np
from scipy import ndimage
from astropy.io import fits

class Image:
    def __init__(self, val, axes=None):
        self.val = val
        self.nx = np.shape(val)[1]
        self.ny = np.shape(val)[0]

        if axes:
            self.x = axes[1]
            self.y = axes[0]

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

    def rotate(self, obs):
        self.val = ndimage.rotate(self.val, 90-obs.PA, reshape=False)

    def sample(self, u, v):
        # sample image at u, v points
        self.square()

    def save(self, obs):
        hdu = fits.PrimaryHDU(self.val, obs.header(self.nx))
        hdu.writeto(obs.modfile, overwrite=True, output_verify='fix')
