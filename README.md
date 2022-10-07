# debris_disk
Produce synthetic images and visibilities of disk structure from a radial, vertical, and scale height functional form. You can produce an image such as

```python
import debris_disk as DD

obs_params = {'nu' : 345.8, # ghz
              'imres' : 0.005 # arcsec / pixel
              'PA' : # degrees
              'distance' : 100} # parsecs

model = DD.Disk(Lstar=1., # L_sun
                Mdust=1e-7, # M_sun
                inc=88.5, # Degrees
                radial_func='powerlaw', # options: gaussian
                radial_params=[2.2],
                disk_edges=[22,42], # au
                sh_func='linear', # options: constant
                sh_params=[0.025],
                vert_func='gaussian',
                obs=obs_params)

model.square()
model.rotate()
model.save('example.fits')
```
