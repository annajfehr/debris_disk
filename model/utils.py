def mk_uv(data):
    '''
    Given the file path to a uv fits file, return u, v, and amplitudes
    '''
    data_vis = fits.open(datafile)

    freq0 = data_vis[0].header['CRVAL4']
    u = (data_vis[0].data['UU']*freq0).astype(np.float64)
    v = (data_vis[0].data['VV']*freq0).astype(np.float64)
    return u, v, data_vis
