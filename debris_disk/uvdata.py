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

    def sample(self, disk=None, val=None, dxy=None, PA=0, dRA=0., dDec=0., resid_dir='resids/', mod_dir='mods/'):
        assert (self.mode != 'mcmc')

        val, dxy = self.prepare_image(disk) 
    
        PA *= np.pi/180
        for i, dataset in enumerate(self.datasets):
            dataset.sample(val, 
                           dxy, 
                           resid_dir+'resid'+str(i)+'.fits',
                           mod_dir+'mod'+str(i)+'.fits',
                           PA=PA,
                           dRA=dRA,
                           dDec=dDec)


    def chi2(self, disk=None, val=None, dxy=None, PA=0., dRA=0., dDec=0.):
        val, dxy = self.prepare_image(disk) 

        PA *= np.pi /180

        chi2 = sum([dataset.chi2(val, dxy, PA=PA, dRA=dRA, dDec=dDec) for dataset in self.datasets])
        return chi2

    def prepare_image(self, disk):
        im = disk.image()
        dxy = im.imres * np.pi /(180*3600) 
        
        im.square()
        val = im.val[::-1,:].copy(order='C')

        min_pixels = int(self._mrs() / dxy)+1

        if min_pixels % 2 != 0:
            min_pixels+=1
        if min_pixels > np.shape(val)[0]:
            val = fill_in(val, min_pixels)

        FOV= np.shape(val)[0] * dxy
        mrs = self._mrs()

        return val, dxy

    def _mrs(self):
        return max([dataset.mrs() for dataset in self.datasets])

class UVDataset:
    def __init__(self, f, mode):
        self.file = f

        data_vis = fits.open(f)

        self.header = data_vis[0].header
        self.freq0 = self.header['CRVAL4']

        data = data_vis[0].data['data']

        self.re = (data[:,0,0,0,:,:,0]).astype(np.float64) 
        self.im = (data[:,0,0,0,:,:,1]).astype(np.float64)
        
        self.w = (data[:,0,0,0,:,:,2]).astype(np.float64)
        
        self.u = (data_vis[0].data['UU']*self.freq0).astype(np.float64).copy(order='C')
        self.v = (data_vis[0].data['VV']*self.freq0).astype(np.float64).copy(order='C')
        
        if mode =='MCMC':
            return

        self.data = data
        self.data_vis = data_vis

    def mrs(self):
        uvdist = np.hypot(self.u, self.v)
        return 0.6 / np.min(uvdist)

    def sample(self, val, dxy, residout='resid.fits', modout='mod.fits', PA=0.,
            dRA=0., dDec=0.):
        model_vis = np.zeros(self.data.shape)

        vis = gd.sampleImage(val, dxy, self.u, self.v, PA=(np.pi/2)+PA)
        for i in range(np.shape(model_vis)[4]):
            model_vis[:, 0, 0, 0, i, 0, 0] = vis.real
            model_vis[:, 0, 0, 0, i, 1, 0] = vis.real
            model_vis[:, 0, 0, 0, i, 0, 1] = vis.imag
            model_vis[:, 0, 0, 0, i, 1, 1] = vis.imag

        outfile = self.data_vis.copy()
        outfile[0].data['data'] = self.data- model_vis
        outfile.writeto(residout, overwrite=True)
        
        model_vis[:, 0, 0, 0, :, 0, 2] =self.data[:,0,0,0,:,0,2]  # weights (XX)
        model_vis[:, 0, 0, 0, :, 1, 2] =self.data[:,0,0,0,:,1,2]  # weights (XX)
        
        outfile[0].data['data'] = model_vis
        outfile.writeto(modout, overwrite=True)
    
    def chi2(self, val, dxy, PA=0., dRA=0., dDec=0.):
        chi2 = 0
        for i in range(np.shape(self.re)[1]):
            re = self.re[:,i,0].copy(order='C')
            im = self.im[:,i,0].copy(order='C')
            w = self.w[:,i,0].copy(order='C')
            chi2 += gd.chi2Image(val, dxy, self.u, self.v, re, im, w,
                                 PA=(np.pi/2)+PA,check=True)
            
            re = self.re[:,i,1].copy(order='C')
            im = self.im[:,i,1].copy(order='C')
            w = self.w[:,i,1].copy(order='C')
            chi2 += gd.chi2Image(val, dxy, self.u, self.v, re, im, w,
                                 PA=(np.pi/2)+PA)
        return chi2

def fill_in(val, min_pixels):
    nx = np.shape(val)[0]
    start = int((min_pixels-nx)/2)+1
    end = int((min_pixels+nx)/2) +1

    new = np.zeros((min_pixels, min_pixels))
    new[start:end, start:end] = val
    return new
