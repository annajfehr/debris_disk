import os
import numpy as np
from astropy.io import fits
import galario.double as gd


class UVDataset:
    def __init__(self, f=None, stored=None, filetype='txt'):
        if stored:
            for key in stored:
                setattr(self, key, stored[key])
            return

        self.files = f
        self.datasets = []
        
        for file in f:
            self.datasets.append(UVData(file, filetype=filetype))

        self._mrs()
        self._resolution()
        self.chans = [element for ds in self.datasets for element in ds.chans]

    def _mrs(self):
        self.mrs = [ds.mrs for ds in self.datasets]

    def _resolution(self):
        self.resolution = [ds.resolution for ds in self.datasets]
    
    def sample(self, val=None, dxy=None, disk=None, modout='mod.txt', PA=0., dRA=0., dDec=0., prefix=''):
        images = disk.ims
        for i, ds in enumerate(self.datasets):
            ds.sample(ims=images[:ds.nchans],
                      dxy=dxy,
                      modout=prefix+str(i)+modout,
                      PA=PA,
                      dRA=dRA,
                      dDec=dDec)
            images = images[ds.nchans:]

    def chi2(self, val=None, dxy=None, disk=None, PA=0., dRA=0., dDec=0., imchecked=False):

        if not disk.mod:
            return np.inf

        chi2=0

        images = disk.ims
        for i, ds in enumerate(self.datasets):
            chi2+=ds.chi2(ims=images[:ds.nchans], dxy=dxy, disk=disk, PA=PA, dRA=dRA, dDec=dDec)
            images = images[ds.nchans:]
        return chi2


class UVData:
    def __init__(self, f=None, stored=None, filetype='txt'):
        if stored:
            for key in stored:
                setattr(self, key, stored[key])
            return
        self.file = f
        self.filetype=filetype

        if filetype == 'txt':
            self.u = []
            self.v = []
            
            self.re = []
            self.im = []

            self.w = []
            data = np.loadtxt(f)
            chans, index = np.unique(data[:, 5],return_index=True)
            self.chans = chans[index.argsort()]

            for chan in self.chans:
                indices = (data[:,5] == chan)
                dat = data[indices]

                self.u.append(dat[:, 0])
                self.v.append(dat[:, 1])

                self.re.append(dat[:, 2])
                self.im.append(dat[:, 3])

                self.w.append(dat[:, 4])

            self.chans *= 100

        if filetype == 'fits':
            data = fits.open(f)[0]
            self.chans = np.arange(0, data.header['NAXIS5']) * data.header['CDELT4'] + data.header['CRVAL4']
            self.u = np.array([data.data['UU'] * chan for chan in self.chans]).astype(np.float64)
            self.v = np.array([data.data['VV'] * chan for chan in self.chans]).astype(np.float64)

            self.re = data.data['data'][:,0,0,:,0,:,0]
            self.im = data.data['data'][:,0,0,:,0,:,1]
            self.w = data.data['data'][:,0,0,:,0,:,2]
            
            self.chans = [2.998e10/chan for chan in self.chans]
            self.data = fits.open(f)

        self._mrs()
        self._resolution()

        self.nchans = len(self.chans)

    def _mrs(self):
        self.mrs = np.empty(len(self.u))
        for i, (u, v) in enumerate(zip(self.u, self.v)):
            self.mrs[i] = find_mrs(u, v)

    def _resolution(self):
        self.resolution = np.empty(len(self.u))
        for i, (u, v) in enumerate(zip(self.u, self.v)):
            self.resolution[i] = find_resolution(u, v)
    
    def sample(self, ims=None, val=None, dxy=None, disk=None, modout='mod.txt', residout=None, PA=0.,
            dRA=0., dDec=0.):

        print(self.file)
        if self.filetype == 'txt':
            model_vis = np.empty((np.sum(len(u) for u in self.u), 6))
        if self.filetype == 'fits':
            model_vis = self.data[0].data['data'].copy()

        PA *= np.pi/180
        dRA *= np.pi /(180*3600)
        dDec *= np.pi/(180*3600)

        if ims==None:
            assert len(disk.ims) == len(self.re)
            ims=disk.ims

        start_loc = 0
        
        
        for i, im in enumerate(ims):
            val, dxy = prepare_image(im, self.mrs[i], self.chans[i])
            ims[i].val = val

            gd.threads(num=1)

            print(i, ":", np.shape(val), dxy, dRA, dDec)
            vis = gd.sampleImage(val,
                    dxy,
                    self.u[i].copy(order='C'), 
                    self.v[i].copy(order='C'), 
                    PA=(np.pi/2+PA),
                    dRA=dRA,
                    dDec=dDec,
                    check=True)
            

            end_loc = start_loc + len(vis.real)

            if self.filetype=='txt':
                mod = [self.u[i], self.v[i], vis.real, vis.imag, self.w[i], [self.chans[i]/100]*len(vis.real)]
                model_vis[start_loc:end_loc] = np.array(mod).T

                start_loc = end_loc

            if self.filetype=='fits':
                model_vis[:, 0, 0, i, 0, 0, 0] = vis.real
                model_vis[:, 0, 0, i, 0, 1, 0] = vis.real
                
                model_vis[:, 0, 0, i, 0, 0, 1] = vis.imag
                model_vis[:, 0, 0, i, 0, 1, 0] = vis.imag

        if self.filetype=='txt':
            np.savetxt(modout, model_vis)

        if self.filetype=='fits': 
            model = self.data.copy()
            model[0].data['data'] = model_vis
            model.writeto(modout, overwrite=True)
            model.close()

    def chi2(self, ims=None, dxy=None, disk=None, PA=0., dRA=0., dDec=0., imchecked=False):
        chi2 = 0
        PA *= np.pi /180
        dRA *= np.pi /(180*3600)
        dDec *= np.pi/(180*3600)

        if ims == None:
            assert len(disk.ims) == len(self.re)
            ims = disk.ims

        for i, im in enumerate(ims):
            val, dxy = prepare_image(im, self.mrs[i], self.chans[i])

            chi2 += gd.chi2Image(val, 
                                 dxy, 
                                 self.u[i].copy(order='C'), 
                                 self.v[i].copy(order='C'), 
                                 self.re[i].copy(order='C'), 
                                 self.im[i].copy(order='C'), 
                                 self.w[i].copy(order='C'), 
                                 PA=(np.pi/2+PA),
                                 dRA=dRA,
                                 dDec=dDec)
        return chi2


def prepare_image(im, mrs, nu):
    dxy = im.imres * np.pi /(180*3600) 

    im.square()
    im.beam_corr(nu, 12)
    val = im.val[::-1,:].copy(order='C')

    min_pixels = int(2 * mrs / dxy)+1

    if min_pixels % 2 != 0:
        min_pixels+=1
    if min_pixels > np.shape(val)[0]:
        val = fill_in(val, min_pixels)

    return val, dxy

def find_mrs(u, v):
    if type(u[0]) == float:
        uvdist = np.hypot(u, v)
    else:
        uvdist = [np.min(np.hypot(up, vp)) for up, vp in zip(u, v)]
    return 0.6 / np.min(uvdist)

def find_resolution(u, v):
    if type(u[0]) == float:
        uvdist = np.hypot(u, v)
    else:
        uvdist = [np.max(np.hypot(up, vp)) for up, vp in zip(u, v)]
    return 1 / (2 * np.max(uvdist))

def fill_in(val, min_pixels):
    nx = np.shape(val)[0]
    start = int((min_pixels-nx)/2)+1
    end = int((min_pixels+nx)/2) +1

    new = np.zeros((min_pixels, min_pixels))
    new[start:end, start:end] = val
    return new

