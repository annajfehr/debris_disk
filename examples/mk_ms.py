import numpy as np

msfile = '../TYC9340/TYC9340.12m.cal.bary.continuum.fav.tav.SMGsub.corrected.ms/'
model_table = 'mod.txt'
datacolumn='DATA'
resid_msfile='resid.ms'
mod_msfile='mod.ms'

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

Vmodel = np.array([Vmodel, Vmodel])
Vmodel_reshaped=np.reshape(Vmodel, shape)

tb.close()

# copy observations before modifying
if resid_msfile:
    os.system('rm -r {}'.format(resid_msfile))
    os.system('cp -r {} {}'.format(msfile, resid_msfile))

    tb.open(resid_msfile, nomodify=False)

    vis_sub=data - Vmodel_reshaped
    # save visibilities
    tb.putcol(datacolumn, vis_sub) # save modified data

    tb.close()

if mod_msfile:
    os.system('rm -r {}'.format(mod_msfile))
    os.system('cp -r {} {}'.format(msfile, mod_msfile))

    tb.open(mod_msfile, nomodify=False)

    vis_sub=Vmodel_reshaped
    # save visibilities
    tb.putcol(datacolumn, vis_sub) # save modified data

    tb.close()
