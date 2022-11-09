import os
import numpy as np
from astropy.io import fits
import galario.double as gd

class UVData:
    def __init__(self, directory, filetype='txt', mode=None):
        '''
        Initialize UVData object from the set of files in directory
        '''
        files = os.listdir(directory)
        self.n = len(files)

        self.directory = directory
        self.mode = mode
        self.datasets = [UVDataset(directory+f, self.mode, filetype) for f in files]

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
        val, dxy = prepare_image(self, disk) 


        chi2 = sum([dataset.chi2(val, dxy, PA=PA, dRA=dRA, dDec=dDec, imchecked=True) for dataset in self.datasets])
        return chi2


    def _mrs(self):
        return max([dataset._mrs() for dataset in self.datasets])

class UVDataset:
    def __init__(self, f, mode, filetype='txt'):
        self.file = f

        if filetype == 'fits':
            data_vis = fits.open(f)

            self.header = data_vis[0].header
            self.freq0 = self.header['CRVAL4']

            data = data_vis[0].data['data']

            self.re = (data[:,0,0,0,:,:,0]).astype(np.float64) 
            self.im = (data[:,0,0,0,:,:,1]).astype(np.float64)
            
            self.w = (data[:,0,0,0,:,:,2]).astype(np.float64)
            
            self.u = (data_vis[0].data['UU']*self.freq0).astype(np.float64).copy(order='C')
            self.v = (data_vis[0].data['VV']*self.freq0).astype(np.float64).copy(order='C')

        if filetype == 'txt':
            data = np.loadtxt(f)


            self.chans = np.sort(np.unique(data[:, 5]))
            
            self.u = []
            self.v = []
            
            self.re = []
            self.im = []

            self.w = []

            for chan in self.chans:
                indices = (data[:,5] == chan)
                dat = data[indices]

                self.u.append(dat[:, 0])
                self.v.append(dat[:, 1])

                self.re.append(dat[:, 2])
                self.im.append(dat[:, 3])

                self.w.append(dat[:, 4])

            self.chans *= 10

        if mode =='mcmc':
            return

        self.data = data
        self.data_vis = data_vis

    def _mrs(self):
        if type(self.u[0]) == float:
            uvdist = np.hypot(self.u, self.v)
        else:
            uvdist = [np.min(np.hypot(u, v)) for u, v in zip(self.u, self.v)]
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
        model_vis[:, 0, 0, 0, :, 1, 2] =self.data[:,0,0,0,:,1,2]  # weights (YY)
        
        outfile[0].data['data'] = model_vis
        outfile.writeto(modout, overwrite=True)
    
    def chi2(self, val=None, dxy=None, disk=None, PA=0., dRA=0., dDec=0., imchecked=False):
        chi2 = 0
        PA *= np.pi /180
        
        assert len(disk.ims) == len(self.re)
        
        for i, im in enumerate(disk.ims):
            val, dxy = prepare_image(im, self.u[i], self.v[i])

            chi2 += gd.chi2Image(val, 
                                 dxy, 
                                 self.u[i].copy(order='C'), 
                                 self.v[i].copy(order='C'), 
                                 self.re[i].copy(order='C'), 
                                 self.im[i].copy(order='C'), 
                                 self.w[i].copy(order='C'), 
                                 PA=(np.pi/2+PA), 
                                 check=True)
        return chi2

        """
        if not imchecked:
            val, dxy = prepare_image(self, disk)

        if len(np.shape(self.re)) == 1:
            for i in range(len(self.re)):
                re = self.re[i].copy(order='C')
                im = self.im[i].copy(order='C')
                w = self.w[i].copy(order='C')
                u = self.u[i].copy(order='C')
                v = self.v[i].copy(order='C')
                chi2 += gd.chi2Image(val, dxy, u, v, re, im, w, PA=(np.pi/2+PA), check=True) 
        else:
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
        """

def prepare_image(im, u, v):
    dxy = im.imres * np.pi /(180*3600) 
    
    im.square()
    val = im.val[::-1,:].copy(order='C')

    min_pixels = int(2 * mrs(u, v) / dxy)+1
    
    if min_pixels % 2 != 0:
        min_pixels+=1
    if min_pixels > np.shape(val)[0]:
        val = fill_in(val, min_pixels)

    return val, dxy

def mrs(u, v):
    if type(u[0]) == float:
        uvdist = np.hypot(u, v)
    else:
        uvdist = [np.min(np.hypot(up, vp)) for up, vp in zip(u, v)]
    return 0.6 / np.min(uvdist)

def fill_in(val, min_pixels):
    nx = np.shape(val)[0]
    start = int((min_pixels-nx)/2)+1
    end = int((min_pixels+nx)/2) +1

    new = np.zeros((min_pixels, min_pixels))
    new[start:end, start:end] = val
    return new
