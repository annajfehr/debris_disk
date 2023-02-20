import os
import numpy as np
from astropy.io import fits
import galario.double as gd

class UVDataset:
    def __init__(self, f=None, mode='mcmc', stored=None, filetype='txt'):
        if stored:
            for key in stored:
                setattr(self, key, stored[key])
            return

        self.file = f

        if filetype == 'txt':
            data = np.loadtxt(f)
            chans, index = np.unique(data[:, 5],return_index=True)
            self.chans = chans[index.argsort()]
            
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

            self.chans *= 100

        self._mrs()
        if mode =='mcmc':
            return

        self.data = data

    def _mrs(self):
        self.mrs = np.empty(len(self.u))
        for i, (u, v) in enumerate(zip(self.u, self.v)):
            self.mrs[i] = find_mrs(u, v)

    def sample(self, val=None, dxy=None, disk=None, residout='resid.txt', modout='mod.txt', PA=0.,
            dRA=0., dDec=0.):
        
        model_vis = np.empty((np.sum(len(u) for u in self.u), 6))
        PA *= np.pi/180

        assert len(disk.ims) == len(self.re)

        start_loc = 0

        for i, im in enumerate(disk.ims):
            val, dxy = prepare_image(im, self.mrs[i], self.chans[i])

            vis = gd.sampleImage(val,
                    dxy,
                    self.u[i].copy(order='C'), 
                    self.v[i].copy(order='C'), 
                    PA=(np.pi/2+PA),
                    dRA=dRA,
                    dDec=dDec)
            end_loc = start_loc + len(vis.real)

            mod = [self.u[i], self.v[i], vis.real, vis.imag, self.w[i], [self.chans[i]]*len(vis.real)]
            model_vis[start_loc:end_loc] = np.array(mod).T

            start_loc = end_loc

        model_vis[:, 5] /= 100

        np.savetxt(modout, model_vis)

    def chi2(self, val=None, dxy=None, disk=None, PA=0., dRA=0., dDec=0., imchecked=False):
        chi2 = 0
        PA *= np.pi /180

        assert len(disk.ims) == len(self.re)
        
        for i, im in enumerate(disk.ims):
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

    def pack(self):
        return ([self.u, self.v, self.re, self.im, self.w, self.chan])

    def make_ms(msfile, model_table, mod_msfile=None, resid_msfile=None, datacolumn='DATA', verbose=True):
        """
        function to subtract model from visibilities and save it as a new ms file
        Parameters
        ==========
        msfile: string, path to ms original file
        new_msfile: string, name of new ms file
        uvtable_filename: string, path to visibility table with model
        datacolumn: string, column to extract (e.g. data or corrected)
        """


        # load model visibilities
        um, vm, Vrealm, Vimagm, wm, lamsm = np.require(np.loadtxt(model_table, unpack=True), requirements='C')
        Vmodel=Vrealm+Vimagm*1j

        # open observations
        tb.open(msfile, nomodify=True)

        # get visibilities
        tb_columns = tb.colnames()
        #print(tb_columns)
        if datacolumn.upper() in tb_columns:
            data = tb.getcol(datacolumn.upper())
        else:
            raise KeyError("datacolumn {} is not available.".format(datacolumn))

        # reshape model visibilities
        shape=np.shape(data)
        nrows, ncol=shape[1], shape[2]
        
        Vmodel_reshaped=np.reshape(Vmodel, (nrows, ncol))
        
        tb.close()
        
        # copy observations before modifying
        if resid_msfile:
            os.system('rm -r {}'.format(resid_msfile))
            os.system('cp -r {} {}'.format(msfile, resid_msfile))

            resid_tb.open(resid_msfile, nomodify=False)

            vis_sub=data - Vmodel_reshaped
            # save visibilities
            resid_tb.putcol(datacolumn, vis_sub) # save modified data

            resid_tb.close()
        
        if mod_msfile:
            os.system('rm -r {}'.format(mod_msfile))
            os.system('cp -r {} {}'.format(msfile, mod_msfile))

            mod_tb.open(mod_msfile, nomodify=False)

            vis_sub=Vmodel_reshaped
            # save visibilities
            mod_tb.putcol(datacolumn, vis_sub) # save modified data

            mod_tb.close()

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

def fill_in(val, min_pixels):
    nx = np.shape(val)[0]
    start = int((min_pixels-nx)/2)+1
    end = int((min_pixels+nx)/2) +1

    new = np.zeros((min_pixels, min_pixels))
    new[start:end, start:end] = val
    return new

