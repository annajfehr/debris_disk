import os
import numpy as np
from astropy.io import fits
import galario.double as gd

class UVData:
    def __init__(self, directory, mode=None):
        '''
        Initialize UVData object from the set of files in directory
        '''
        files = os.listdir(directory)
        self.n = len(files)

        self.directory = directory
        self.mode = mode
        self.datasets = [UVDataset(directory+f, self.mode) for f in files]
    
    def sample(self, val, dxy, resid_dir='resids/', mod_dir='mods/'):
        assert (self.mode != 'MCMC')
        for i, dataset in enumerate(self.datasets):
            dataset.sample(val, 
                           dxy, 
                           resid_dir+'resid'+str(i)+'.fits',
                           mod_dir+'mod'+str(i)+'.fits')


    def chi2(self, val, dxy):
        chi2 = sum([dataset.chi2(val, dxy) for dataset in self.datasets])
        return chi2

class UVDataset:
    def __init__(self, f, mode):
        self.file = f

        data_vis = fits.open(f)

        self.header = data_vis[0].header
        self.freq0 = self.header['CRVAL4']
        
        data = data_vis[0].data['data']

        self.re = (data[:,0,0,0,:,0,0]).astype(np.float64).copy(order='C')
        self.im = (data[:,0,0,0,:,0,1]).astype(np.float64).copy(order='c')
        
        self.w = (data[:,0,0,0,:,0,2]).astype(np.float64).copy(order='C')
        self.u = (data_vis[0].data['UU']*self.freq0).astype(np.float64).copy(order='C')
        self.v = (data_vis[0].data['VV']*self.freq0).astype(np.float64).copy(order='C')
        
        if mode =='MCMC':
            return

        self.data = data
        self.data_vis = data_vis

    def sample(self, val, dxy, residout='resid.fits', modout='mod.fits'):
        model_vis = np.zeros(self.data.shape)

        vis = gd.sampleImage(val, dxy, self.u, self.v)
        for i in range(np.shape(model_vis)[4]):
            model_vis[:, 0, 0, 0, i, 0, 0] = vis.real
            model_vis[:, 0, 0, 0, i, 1, 0] = vis.real
            model_vis[:, 0, 0, 0, i, 0, 1] = vis.imag
            model_vis[:, 0, 0, 0, i, 1, 1] = vis.imag

        outfile = self.data_vis.copy()
        outfile[0].data['data'] = self.data- model_vis
        outfile.writeto(residout, overwrite=True)
        
        model_vis[:, 0, 0, 0, :, :, 2] = self.w  # weights (XX)
        #model_vis[:, 0, 0, 0, :, 1, 2] = self.w  # weights (YY)
        outfile[0].data['data'] = model_vis
        outfile.writeto(modout, overwrite=True)
    
    def chi2(self, val, dxy):
        chi2 = 0
        for i in range(np.shape(self.re)[1]):
            re = self.re[:,i].copy(order='C')
            im = self.im[:,i].copy(order='C')
            w = self.w[:,i].copy(order='C')
            chi2 += gd.chi2Image(val, dxy, self.u, self.v, re, im, w)
        return chi2
