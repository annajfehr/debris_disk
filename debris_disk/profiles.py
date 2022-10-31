import numpy as np

class powerlaw:
    def val(r, alpha, Rin, Rout, lin=0.01, lout=0.01):
        return (r/Rin)**alpha * (1+np.tanh((r-Rin)/lin)) * \
                (1+np.tanh((Rout-r)/lout))/4
    def limits(alpha, Rin, Rout, lin=0.01, lout=0.01):
        fac = 3.451 # 99.9 percentile
        rmin = max(0, Rin - lin*fac)
        rmax = Rout + lout*fac
        return [rmin, rmax]
    def params():
        return {'r' : None,
                'alpha' : None,
                'Rin' : None,
                'Rout' : None,
                'lin' : 0.01,
                'lout' : 0.01}


class double_powerlaw:
    def val(r, rc, alpha_in, alpha_out, gamma):
        value = ((r/rc)**(-alpha_in*gamma) + \
                (r/rc)**(-alpha_out*gamma))**(-1/gamma)
        return value / np.max(value)
    def limits(rc, alpha_in, alpha_out, gamma):
        assert alpha_in > 0, "alpha_in must be positive to find inner edge"
        assert alpha_out < 0, "alpha_out must be negative to find outer edge"
        rmax = rc + 5.28e15 + 2.94*(alpha_out**2) + 58.5*alpha_out
        return [0, rmax]

class triple_powerlaw:
    def val(r, Rin, Rout, alpha_in, alpha_mid, alpha_out, gamma_in, gamma_out):
        S1=((r/Rin)**(-alpha_in*gamma_in) + \
                (r/Rin)**(-alpha_mid*gamma_in))**(-1./gamma_in)
        S2=((r/Rout)**(-alpha_mid*gamma_out) + \
                (r/Rout)**(-alpha_out*gamma_out))**(-1./gamma_out)
        return S1*S2 / (Rin/Rout)**alpha_mid
    def limits(Rin, Rout, alpha_in, alpha_mid, alpha_out, gamma_in, gamma_out):
        assert alpha_in > 0, "alpha_in must be positive to find inner edge"
        assert alpha_out < 0, "alpha_out must be negative to find outer edge"
        rmin = Rin - (1000/((alpha_in + 8.61)**(5/4)))
        rmax = Rout + (1000/((alpha_out +8.61)**(5/4)))
        return [min(rmin, 0), rmax]

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
    def val(x, sigma, x0=0):
        return np.exp(-((x-x0)**2/(2*sigma**2)))
    def norm(sigma, x0=0):
        return sigma * np.sqrt(2*np.pi)
    def limits(sigma, x0=0):
        fac = 2.628 # 99.9 percentile
        rmin = max(0, x0 - sigma * fac)
        rmax = x0 + sigma * fac
        return [rmin, rmax]

class lorentzian:
    def val(x, gamma, b=0):
        return 1/((x-b)**2 + (gamma/2)**2) 
    def norm(gamma, b=0):
        return gamma/(2*np.pi)
    def limits(gamma, b=0):
        fac = 15.8 # 99.9 percentile
        rmin = max(0, b - gamma * fac)
        rmax = b + gamma * fac
        return [rmin, rmax]
