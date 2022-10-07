import numpy as np
from scipy import ndimage
from astropy.io import fits

class Image:
    def __init__(self, val, imres=None, axes=None):
        self.val = val
        self.nx = np.shape(val)[1]
        self.ny = np.shape(val)[0]

        if imres:
            self.imres = imres

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

    def sample(self, vis_temp, u, v, obs):

        data = data_vis[0].data['data']
        re = ((data[:,0,0,0,:,0,0]).astype(np.float64).copy(order='C'))
        im = ((data[:,0,0,0,:,0,1]).astype(np.float64).copy(order='C'))
        w = ((data[:,0,0,0,:,0,2]).astype(np.float64).copy(order='C'))
        model_vis = np.zeros(data.shape)
        
        self.square()
        
        if not saveim:
            return chi2Image(self.val, self.imres*np.pi/(180*3600), u, v)
        
        vis = sampleImage(self.val, self.imres*np.pi/(180*3600), u, v)
        
        for i in range(np.shape(model_vis)[4]):
            model_vis[:, 0, 0, 0, i, 0, 0] = vis.real
            model_vis[:, 0, 0, 0, i, 1, 0] = vis.real
            model_vis[:, 0, 0, 0, i, 0, 1] = vis.imag
            model_vis[:, 0, 0, 0, i, 1, 1] = vis.imag

        data_vis[0].data['data'] = data_vis[0].data['data'] - model_vis
        data_vis.writeto(residout, overwrite=True)
        
        model_vis[:, 0, 0, 0, :, 0, 2] = data[:, 0, 0, 0, :, 0, 2]  # weights (XX)
        model_vis[:, 0, 0, 0, :, 1, 2] = data[:, 0, 0, 0, :, 1, 2]  # weights (YY)
        data_vis[0].data['data'] = model_vis
        data_vis.writeto(modelout, overwrite=True)
        data_vis.close()

    def save(self, obs, outfile):
        hdu = fits.PrimaryHDU(self.val, obs.header(self.nx))
        hdu.writeto(outfile, overwrite=True, output_verify='fix')
