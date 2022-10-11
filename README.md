# debris_disk
Produce synthetic images and visibilities of disk structure from a radial, vertical, and scale height functional form. You can produce an image such as

```python
import debris_disk as DD

obs_params = {'nu' : 345.8, # ghz
              'imres' : 0.005 # arcsec / pixel
              'PA' : # degrees
              'distance' : 100} # parsecs

rad_params = {'alpha' : 2.2,
              'Rin' : 22 * DD.constants.AU,
              'Rout' : 42 * DD.constants.AU}

model = DD.Disk(Lstar=1., # L_sun
                sigma_crit=1e-7, # g/m^2
                inc=88.5, # Degrees
                radial_func='powerlaw', # options: double_powerlaw,
                                        # triple_powerlaw, gaussian
                radial_params=rad_params,
                aspect_ratio = 0.025,
                vert_func='gaussian',
                obs=obs_params)

model.square()
model.rotate()
model.save('example.fits')
```

From a disk model, produce synthetic visibilities as

```python
vis=DD.UVDataset('data_directory') 
vis.sample(model)
```

where 'data_directory' is the filepath to a directory containing only uvfits
files.

From a disk model, calculate a $`\chi^2`$ value from visibilities like

```python
vis.chi2(mod)
```
