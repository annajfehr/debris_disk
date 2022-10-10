import numpy as np

class powerlaw:
    def val(x, p):
        return x**p
    def norm(p, Rin, Rout):
        return (1 / (p + 2)) * ((Rout**(p+2)) - (Rin**(p+2)))
  
class powerlaw_edges:
    def val(r, alpha, Rin, Rout, lin=0.01, lout=0.01):
        return (r/Rin)**alpha * (1+np.tanh((r-Rin)/lin)) * \
                (1+np.tanh((Rout-r)/lout))/4

    def limits(alpha, Rin, Rout, lin=0.01, lout=0.01):
        fac = 3.451 # 99.9 percentile
        rmin = max(0, Rin - lin*fac)
        rmax = Rout + lout*fac
        return [rmin, rmax]

class constant:
    def val(x, c):
        return c * x / x
    def norm(c, Rin, Rout):
        return c * (Rout - Rin)

class linear:
    def val(x, slope):
        return slope * x
    def norm(slope, Rin, Rout):
        return slope*(Rout**2-Rin**2)/2

class gaussian:
    def val(x, c, b=0):
        return np.exp(-((x-b)/c)**2)
    def norm(c, b=0):
        return c * np.sqrt(np.pi)
    def limits(c, b=0):
        fac = 2.628 # 99.9 percentile
        rmin = max(0, b - c * fac)
        rmax = b + c * fac
        return [rmin, rmax]
