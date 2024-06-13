import os
import numpy as np
from astropy.io import fits
import galario.double as gd
import debris_disk.constants as const
from debris_disk import profiles

class UVDataset:
    def __init__(self, f=None, stored=None, filetype='txt', freq=3.38225e11): # old freq was in as 3.380857111374e11
        # frequency is in Hz
        if stored:
            for key in stored:
                setattr(self, key, stored[key])
            return

        self.files = f
        self.datasets = []
        
        for file in f:
            self.datasets.append(UVData(file, filetype=filetype))

        self._mrs() # maximum recoverable scale for the whole dataset
        self._resolution()
        self.chans = [ds.chans for ds in self.datasets] # yeah so unfortunately in the code's current form, this could be either lambda [cm] or freq [Hz]
        self.freq = freq # frequency of obs in Hz

    def _mrs(self):
        self.mrs = [ds.mrs for ds in self.datasets]

    def _resolution(self):
        self.resolution = [ds.resolution for ds in self.datasets]
    
    def sample(self, disk=None, ims=None, modout='mod.txt', PA=0., dRA=0., dDec=0., F_star=0, prefix=''):
        if ims:
            images = ims
        else:
            images = disk.ims
        for i, ds in enumerate(self.datasets):
            ds.sample(ims=[images[0]],
                      modout=prefix+str(i)+modout,
                      PA=PA,
                      dRA=dRA,
                      dDec=dDec,
                      F_star=F_star)
            images = images[1:]

    def chi2(self, val=None, dxy=None, disk=None, PA=0., dRA=0., dDec=0., F_star=0,  imchecked=False):

        if not disk.mod:
            return np.inf
        
        chi2=0

        images = disk.ims # the images we generated when the model Disk was initialized
        
        for i, ds in enumerate(self.datasets): # so ds is a single UVData object
            chi2+=ds.chi2(ims=[images[0]], dxy=dxy, disk=disk, PA=PA, dRA=dRA, dDec=dDec, F_star=F_star)
            images = images[1:] # okay either have this line and images[0], or omit this line and have images[i]
        return chi2


class UVData:
    def __init__(self, f=None, stored=None, filetype='txt', freq=3.38225e11): # old freq was in as 3.380857111374e11
        if stored:
            for key in stored:
                setattr(self, key, stored[key])
            return
        self.file = f
        self.filetype=filetype
        self.freq = freq

        if filetype == 'txt': # how we actually read in data to begin modeling
            self.u, self.v, self.real, self.imag, self.w, self.lams = np.require(np.loadtxt(f, unpack=True), requirements='C')
            self.chans  = np.unique(self.lams)[0] # the lambda values in [m], actually this is only one of them. Don't wanna change it and break everything. RETURN TO THIS LATER
            self.chans *= 100 # the lambda values in [cm]

        if filetype == 'fits':
            print("Warning, you're in the fits zone of UVData in uvdata.py! We haven't checked for unit issues here!")
            data = fits.open(f)[0]
            self.chans = np.arange(0, data.header['NAXIS5']) * data.header['CDELT4'] + data.header['CRVAL4']
            self.u = np.array([data.data['UU'] * chan for chan in self.chans]).astype(np.float64)
            self.v = np.array([data.data['VV'] * chan for chan in self.chans]).astype(np.float64)

            self.re = data.data['data'][:,0,0,:,0,:,0]
            self.re = np.swapaxes(self.re, 0, 1).astype(np.float64)
            self.re = self.re.byteswap().newbyteorder().squeeze()

            self.im = data.data['data'][:,0,0,:,0,:,1]
            self.im = self.im.byteswap().newbyteorder().squeeze()
            self.im = np.swapaxes(self.im, 0, 1).astype(np.float64)

            self.w = data.data['data'][:,0,0,:,0,:,2]
            self.w = self.w.byteswap().newbyteorder().squeeze()
            self.w = np.swapaxes(self.w, 0, 1).astype(np.float64)
            
            self.chans = [const.c/chan for chan in self.chans] # this assumes that chan in self.chans is in cm
            self.data = fits.open(f)

        self._mrs() #maximum recoverable scale for this UVData
        self._resolution()

        #self.nchans = len(self.chans)

    def _mrs(self): # get the maximum recoverable scale for this set of uv data
        self.mrs = find_mrs(self.u, self.v)

    def _resolution(self):
        self.resolution = find_resolution(self.u, self.v)
    
    def sample(self, ims=None, dxy=None, disk=None, modout='mod', PA=0., dRA=0., dDec=0., F_star=0, imchecked=False):
        PA *= np.pi /180 # PA is now in RADIANS!
        dRA *= np.pi /(180*3600) 
        dDec *= np.pi/(180*3600) 
        gd.threads(num=1)
        Vstar=F_star*np.exp(2*np.pi*1j*(self.u*dRA +self.v*dDec))

        for i, im in enumerate(ims):
            val, dxy = prepare_image(im, self.mrs, self.freq)#self.chans)
            if self.filetype=='txt':
                # get complex array of model visibilities
                Vmodel = gd.sampleImage(val, # [Jy/pixel]
                                    dxy, # [rad]
                                    self.u, self.v,
                                    dRA=dRA, 
                                    dDec=dDec, 
                                    PA=PA+np.pi/2.0,
                                    origin='lower')
                # add star in visibility space
                Vmodel=Vmodel+Vstar
                print("saving model!")
                np.savetxt('./'+'{}.txt'.format(modout),
                           np.column_stack([self.u, self.v, Vmodel.real, Vmodel.imag, self.w, self.lams]),
                           fmt='%10.6e', delimiter='\t',
                           header='Model {}.\nwavelength[m] = {}\nColumns:\tu[lambda]\tv[lambda]\tRe(V)[Jy]\tIm(V)[Jy]\tweight\tlambda[m]'.format(modout, np.mean(self.lams)))


    def chi2(self, ims=None, dxy=None, disk=None, PA=0., dRA=0., dDec=0., F_star=0, imchecked=False):
        chi2 = 0
        PA *= np.pi /180 # PA IS IN RADIANS
        dRA *= np.pi /(180*3600) 
        dDec *= np.pi/(180*3600) 

        if ims == None:
            assert len(disk.ims) == len(self.re)
            ims = disk.ims # get the model images from the model disk

        gd.threads(num=1)
        Vstar=F_star*np.exp(2*np.pi*1j*(self.u*dRA +self.v*dDec))

        for i, im in enumerate(ims):
            val, dxy = prepare_image(im, self.mrs, self.freq)#self.chans) # apply primary beam correction and get dxy in radians
            if self.filetype=='txt':
                chi2 += gd.chi2Image(val, # model image, in units of [Jy/pixel]
                                     dxy, # size of square image cell/pixel in [rad] (generated in prepare_image())
                                     self.u, # u visibility coordinates (in units of wavelength, as read in from the text files)
                                     self.v, # v visibility coordinates (in units of wavelength, as read in from the text files)
                                     self.real-Vstar.real, # real component of vis [Jy], as read in from the text files
                                     self.imag-Vstar.imag, # imaginary component of vis [Jy], as read in from the text files
                                     self.w, # visibility weights, as read in from the text files
                                     PA=(np.pi/2+PA),
                                     dRA=dRA,
                                     dDec=dDec,
                                     origin='lower')
            if self.filetype=='fits':
                re = self.re[i,:,0].byteswap().newbyteorder().squeeze()
                imag = self.im[i,:,0]
                w = self.w[i,:,0]
                chi2 += gd.chi2Image(val, 
                                     dxy, 
                                     self.u[i].copy(order='C'), 
                                     self.v[i].copy(order='C'), 
                                     re.copy(order='C'), 
                                     imag.copy(order='C'), 
                                     w.copy(order='C'), 
                                     PA=(np.pi/2+PA),
                                     dRA=dRA, #add stuff from seba here?
                                     dDec=dDec,
                                     origin='lower') #add stuff from seba here?
                re = self.re[i,:,1].byteswap().newbyteorder().squeeze()
                imag = self.im[i,:,1]
                w = self.w[i,:,1]
                chi2 += gd.chi2Image(val, 
                                     dxy, 
                                     self.u[i].copy(order='C'), 
                                     self.v[i].copy(order='C'), 
                                     re.copy(order='C'), 
                                     imag.copy(order='C'), 
                                     w.copy(order='C'), 
                                     PA=(np.pi/2+PA),
                                     dRA=dRA, #add stuff from seba here?
                                     dDec=dDec,
                                     origin='lower') #add stuff from seba here?

        return chi2


def prepare_image(im, mrs, nu):
    dxy = im.imres * np.pi /(180*3600) # imres in arcsec to dxy [rad]
    
    im.square()
    im.beam_corr(nu, 12)
    val = im.val[::-1,:].copy(order='C')
    min_pixels = int(2 * mrs / dxy)+1

    if min_pixels % 2 != 0:
        min_pixels+=1
    if min_pixels > np.shape(val)[0]:
        val = fill_in(val, min_pixels)

    return val, dxy

def prepare_val(val, dxy, mrs, nu, imres):
    # essentially copy of beam corr from image.py, but modded to not require an Image object
    lamb = (const.c/100) / nu # wavelength in m
    start = int(val[0].size/2) # starting point of the meshgrid (so that it ends up centered at 0)
    xvals = imres*(np.linspace(-start, start-1, val[0].size)) # from -rmax to rmax
    yvals = imres*(np.linspace(-start, start-1, val[0].size)) # from -rmax to rmax
    xx, yy = np.meshgrid(xvals, yvals) # mesh grid of coords, centered on center of image (i.e. goes from -rmax to rmax along both axes)
    dst = np.sqrt(xx**2+yy**2) # distance of each coord from center of meshgrid
    D=12 # 12 m antennae
    sigma = const.rad * 1.13 * lamb / D / (2 * np.sqrt(2 * np.log(2))) # 1.13 lamda/D gives the angular FWHM of the field of view, dividing by (2 sqrt(2np.log(2))) links to standard deviation sigma, multiplying by 206265 converts radians to "
    beam = profiles.gaussian.val(dst, sigma) # both dst and sigma in arcseconds, make a gaussian for the primary beam
    val *= beam # multiply image values by beam
    
    val = val[::-1,:].copy(order='C')
    min_pixels = int(2 * mrs / dxy)+1

    if min_pixels % 2 != 0:
        min_pixels+=1
    if min_pixels > np.shape(val)[0]:
        val = fill_in(val, min_pixels)

    return val

def find_mrs(u, v): # maximum recoverable scale (comes from shortest baseline)
    if type(u[0]) == float:
        uvdist = np.hypot(u, v)
    else:
        uvdist = [np.min(np.hypot(up, vp)) for up, vp in zip(u, v)]
    return 0.6 / np.min(uvdist) # see Eqn 3.28 in the ALMA technical handbook (Cycle 11)

def find_resolution(u, v): # angular resolution (comes from longest baseline)
    if type(u[0]) == float:
        uvdist = np.hypot(u, v)
    else:
        uvdist = [np.max(np.hypot(up, vp)) for up, vp in zip(u, v)]
    return 1 / (2 * np.max(uvdist)) # see Eqn 3.27 in the ALMA technical handbook (Cycle 11)

def fill_in(val, min_pixels):
    nx = np.shape(val)[0]
    start = int((min_pixels-nx)/2)+1
    end = int((min_pixels+nx)/2) +1

    new = np.zeros((min_pixels, min_pixels))
    new[start:end, start:end] = val
    return new


def chiSq(datafile, modfile, fileout=None, dxy=None, dRA=0, dDec=0, PA=0, F_star=0, residual=False, filetype='fits', imres=0.005, freq=3.38225e11): # old freq was in as 3.380857111374e11
    dRA *= np.pi/180/3600 
    dDec *= np.pi/180/3600 
    PA *= np.pi/180 
    modfile=fits.open(modfile+str('.fits'))

    if type(datafile) == list:
        cs = 0
        for i, df in enumerate(datafile):
            if fileout: 
                if filetype=='fits':
                    cs += fits_chiSq(df, modfile, str(i)+fileout, dxy, dRA, dDec, PA, F_star, residual)
                if filetype=='txt':
                    cs += txt_chiSq(df, modfile, str(i)+fileout, dxy, dRA, dDec, PA, F_star,  residual, imres, freq)
            else:
                if filetype=='fits':
                    cs += fits_chiSq(df, modfile, fileout, dxy, dRA, dDec, PA, F_star, residual)
                if filetype=='txt':
                    cs += txt_chiSq(df, modfile, fileout, dxy, dRA, dDec, PA, F_star, residual, imres, freq)
        return cs
    else:
        if filetype=='fits':
            return fits_chiSq(df, modfile, fileout, dxy, dRA, dDec, PA, F_star, residual)
        if filetype=='txt':
            return txt_chiSq(df, modfile, fileout, dxy, dRA, dDec, PA, F_star, residual)

def txt_chiSq(datafile, modfile, fileout=None, dxy=None, dRA=0, dDec=0, PA=0, F_star=0, residual=False, imres=0.005, freq=3.38225e11): # old freq was in as 3.380857111374e11
    image=modfile[0].data.astype('double') # important for Galario sometimes
    dpix_deg=modfile[0].header['CDELT2']
    dpix_rad=dpix_deg*np.pi/180.

    u, v, Vreal, Vimag, w, lams = np.require(np.loadtxt(datafile, unpack=True), requirements='C')
    mrs = find_mrs(u, v)

    image = prepare_val(image, dpix_rad, mrs, freq, imres)
    Vstar=F_star*np.exp(2*np.pi*1j*(u*dRA +v*dDec))

    if fileout:
        Vmodel = gd.sampleImage(image, # must be in Jy/pixel
                            dpix_rad, # must be in radians
                            u, v, # must be dimensionless (in units of observing wavelength)
                            dRA=dRA, 
                            dDec=dDec, 
                            PA=PA+np.pi/2.0,
                            origin='lower')
        # add star in visibility space
        Vmodel.real+=Vstar.real
        Vmodel.imag+=Vstar.imag
        #Vmodel=Vmodel+Vstar

        np.savetxt('./'+'model_{}.txt'.format(fileout),
                   np.column_stack([u, v, Vmodel.real, Vmodel.imag, w, lams]),
                   fmt='%10.6e', delimiter='\t',
                   header='Model {}.\nwavelength[m] = {}\nColumns:\tu[lambda]\tv[lambda]\tRe(V)[Jy]\tIm(V)[Jy]\tweight\tlambda[m]'.format(fileout, np.mean(lams)))
    
    
    # compute chi2
    # subtract fstar first from observed visibilities
    chi2 = gd.chi2Image(image, dpix_rad, u, v,
                     Vreal-Vstar.real,
                     Vimag-Vstar.imag,
                     w,
                     dRA=dRA,
                     dDec=dDec,
                     PA=PA+np.pi/2.0,
                     origin='lower' )
    return chi2

def fits_chiSq(datafile, modfile, fileout=None, dxy=None, dRA=0, dDec=0, PA=0, F_star=0, residual=False):
        model_fits = fits.open(modfile) # I now open the model fits file

        if dxy == None:
            dxy = model_fits[0].header['CDELT2']
            dxy *= np.pi/180
            
        model = model_fits[0].data           # and only select the data. Important to do this
        model = model.byteswap().newbyteorder().squeeze() 
        model = model[:,::-1] 
        model_cor = model.copy(order='C') 
        model_fits.close() # always close your fits files when you are done with them!! :) 

        data_vis = fits.open(datafile) # open the data visibilities file
        data_shape = data_vis[0].data['data'].shape # obtain the SHAPE of the data visibilities, which will serve as
        delta_freq = data_vis[0].header['CDELT4'] # This is the channel width. I don't think I actually used this quantity in the end, lol
        freq0 = data_vis[0].header['CRVAL4'] # central frequency of observation used for making the visibilties
        n_spw = data_vis[0].header['NAXIS5'] # number of spectral windows taken from the header of the file
        model_vis = np.zeros(data_shape) # I now create an array of zeroes with the same shape as our "skeleton"

        data = data_vis[0].data['data'] # and in this line I pull out the data

        chi = 0

    #new code (Saad)
        freq_start = freq0 - (n_spw - 1.0) / 2.0 * delta_freq

        Vstar=F_star*np.exp(2*np.pi*1j*(u*dRA +v*dDec))
        for i in range(n_spw):
                freq = freq_start + i * delta_freq
                u, v = (data_vis[0].data['UU'] * freq).astype(np.float64), (data_vis[0].data['VV'] * freq).astype(np.float64)
                mrs = find_mrs(u, v)
                image = prepare_val(model_cor, dxy, mrs, 3.3e11)

                image = np.require(image, requirements='C') # and this is from the galario documentation -- you get some stupid

                vis = gd.sampleImage(image[:,:], dxy, u, v, dRA = dRA, dDec = dDec, PA=PA+np.pi/2.0, origin='lower') 
                vis = vis + Vstar

                model_vis[:,0,0,i,0,0,0] = vis.real
                model_vis[:,0,0,i,0,1,0] = vis.real # here I throw the created model visibilities into corresponding 
                model_vis[:,0,0,i,0,0,1] = vis.imag
                model_vis[:,0,0,i,0,1,1] = vis.imag # same thing but imaginary weights now (XX and YY polarisations)
                chi += ((vis.real - data[:,0,0,i,0,0,0])**2 * data[:,0,0,i,0,0,2]).sum() + ((vis.imag - data[:,0,0,i,0,0,1])**2 * data[:,0,0,i,0,0,2]).sum() + ((vis.real - data[:,0,0,i,0,1,0])**2 * data[:,0,0,i,0,1,2]).sum() + ((vis.imag - data[:,0,0,i,0,1,1])**2 * data[:,0,0,i,0,1,2]).sum()

        if fileout==None:
            return chi

        if residual == False:
                model_vis[:,0,0,:,0,0,2] = data[:,0,0,:,0,0,2] # now I copy the weights from the data (XX) and paste them
                model_vis[:,0,0,:,0,1,2] = data[:,0,0,:,0,1,2] # same things w.r.t. weights (YY)


                data_vis[0].data['data'] = model_vis
                data_vis.writeto(fileout, overwrite=True)
                data_vis.close()

        if residual == True:

            data_vis[0].data['data'] = data_vis[0].data['data'] - model_vis # subtract here

            data_vis.writeto(fileout, overwrite=True) # create your fits file with name via parameter "fileout"

            data_vis.close() # close your fits files!

        return chi

def chiSqStack(datafile, modfile, fileout=None, dxy=None, dRA=0, dDec=0, residual=False):
        if type(datafile) == list:
            datafile=datafile[0]

        data_vis = fits.open(datafile) # open the data visibilities file
        data_shape = data_vis[0].data['data'].shape # obtain the SHAPE of the data visibilities, which will serve as
        delta_freq = data_vis[0].header['CDELT4'] # This is the channel width. I don't think I actually used this quantity in the end, lol
        freq0 = data_vis[0].header['CRVAL4'] # central frequency of observation used for making the visibilties
        n_spw = data_vis[0].header['NAXIS5'] # number of spectral windows taken from the header of the file
        model_vis = np.zeros(data_shape) # I now create an array of zeroes with the same shape as our "skeleton"

        data = data_vis[0].data['data'] # and in this line I pull out the data

        chi = 0

        freq_start = freq0 - (n_spw - 1.0) / 2.0 * delta_freq

        for i in range(n_spw):
                modfile_name = modfile + str('.fits') +str(i)+str('.fits') # turn your "modfile" name into a string with a .fits on it so that you
                                                      # can manipulate your model fits file

                model_fits = fits.open(modfile_name) # I now open the model fits file
                if dxy == None:
                    dxy = model_fits[0].header['CDELT2']

                model = model_fits[0].data           # and only select the data. Important to do this
                model = model.byteswap().newbyteorder().squeeze() # I have no clue why you need to do this, but you do,
                                                                  # at least for the data I had. Otherwise you get a really
                                                                  # strange error.
                model = model[:,::-1] # I believe this flips the data on one axis. Again this may not be important for you,
                                      # but for whatever reason I had to do this because my model was flipped compared to data
                model_cor = model.copy(order='C') # This was also thrown in due to some error. May not be necessary. This might
                                              # fix an error that you get though 
                model_fits.close() # always close your fits files when you are done with them!! :) 
                
                freq = freq_start + i * delta_freq
                u, v = (data_vis[0].data['UU'] * freq).astype(np.float64), (data_vis[0].data['VV'] * freq).astype(np.float64)
                foo = model_cor # this is just the corrected model image which i renamed for some weird reason
                
                vis = gd.sampleImage(model_cor[:,:], dxy, u, v, dRA = dRA, dDec = dDec, PA=PA+np.pi/2.0, origin='lower') #add stuff from seba here?

                model_vis[:,0,0,i,0,0,0] = vis.real
                model_vis[:,0,0,i,0,1,0] = vis.real # here I throw the created model visibilities into corresponding 
                model_vis[:,0,0,i,0,0,1] = vis.imag
                model_vis[:,0,0,i,0,1,1] = vis.imag # same thing but imaginary weights now (XX and YY polarisations)
                chi += ((vis.real - data[:,0,0,i,0,0,0])**2 * data[:,0,0,i,0,0,2]).sum() + ((vis.imag - data[:,0,0,i,0,0,1])**2 * data[:,0,0,i,0,0,2]).sum() + ((vis.real - data[:,0,0,i,0,1,0])**2 * data[:,0,0,i,0,1,2]).sum() + ((vis.imag - data[:,0,0,i,0,1,1])**2 * data[:,0,0,i,0,1,2]).sum()

        if fileout==None:
            return chi

        if residual == False:
                model_vis[:,0,0,:,0,0,2] = data[:,0,0,:,0,0,2] # now I copy the weights from the data (XX) and paste them
                model_vis[:,0,0,:,0,1,2] = data[:,0,0,:,0,1,2] # same things w.r.t. weights (YY)


                data_vis[0].data['data'] = model_vis
                data_vis.writeto(fileout, overwrite=True)
                data_vis.close()

        if residual == True:

            data_vis[0].data['data'] = data_vis[0].data['data'] - model_vis # subtract here

            data_vis.writeto(fileout, overwrite=True) # create your fits file with name via parameter "fileout"

            data_vis.close() # close your fits files!

        return chi
