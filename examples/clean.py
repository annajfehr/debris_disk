tclean(vis='mod.ms/',
       imagename='mod',
       cell=0.04,
       imsize=[1024,1024],
       specmode='mfs',
       weighting='briggs',
       robust=0.5,
       uvtaper='',
       niter=10000,
       interactive=False,
       pbcor=True)

tclean(vis='resid.ms/',
       imagename='resid',
       cell=0.04,
       imsize=[1024,1024],
       specmode='mfs',
       weighting='briggs',
       robust=0.5,
       uvtaper='',
       niter=10000,
       interactive=False,
       pbcor=True)

