from astropy import constants as const

AU      = const.au.cgs.value       # - astronomical unit (cm)
Rsun    = const.R_sun.cgs.value    # - radius of the sun (cm)
c       = const.c.cgs.value        # - speed of light (cm/s)
h       = const.h.cgs.value        # - Planck's constant (erg/s)
kB      = const.k_B.cgs.value      # - Boltzmann's constant (erg/K)
sigmaB  = const.sigma_sb.cgs.value # - Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
pc      = const.pc.cgs.value       # - parsec (cm)
Jy      = 1.e23                    # - cgs flux density (Janskys)
Lsun    = const.L_sun.cgs.value    # - luminosity of the sun (ergs)
Mearth  = const.M_earth.cgs.value  # - mass of the earth (g)
mh      = const.m_p.cgs.value      # - proton mass (g)
Da      = mh                       # - atmoic mass unit (g)
Msun    = const.M_sun.cgs.value    # - solar mass (g)
G       = const.G.cgs.value        # - gravitational constant (cm^3/g/s^2)
rad     = 206264.806               # - radian to arcsecond conversion
kms     = 1e5                      # - convert km/s to cm/s
GHz     = 1e9                      # - convert from GHz to Hz
mCO     = 12.011+15.999            # - CO molecular weight
mDCO    = mCO+2.014                # - HCO molecular weight
mu      = 2.37                     # - gas mean molecular weight
m0      = mu*mh                    # - gas mean molecular opacity
Hnuctog = 0.706*mu                 # - H nuclei abundance fraction (H nuclei:gas)
sc      = 1.59e21                  # - Av --> H column density (C. Qi 08,11)
H2tog   = 0.8                      # - H2 abundance fraction (H2:gas)
Tco     = 19.                      # - freeze out
sigphot = 0.79*sc                  # - photo-dissociation column
Ghz     = 1e9                      # - Ghz (hz)
beta    = 1
c       = const.c.cgs.value        # - speed of light (cm/s)
h       = const.h.cgs.value        # - Planck's constant (erg/s)
kB      = const.k_B.cgs.value      # - Boltzmann's constant (erg/K)