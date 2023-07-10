import numpy as np
from scipy.special import erf

class powerlaw:
    def val(r, alpha, Rin, Rout, lin=0.01, lout=0.01):
        return (r/Rin)**alpha * (1+np.tanh((r-Rin)/lin)) * \
                (1+np.tanh((Rout-r)/lout))/4
    
    def limits(alpha, Rin, Rout, lin=0.01, lout=0.01):
        fac = 3.451 *1.496e13 # 99.99 percentile
        rmin = max(0, Rin - lin*fac)
        rmax = Rout + lout*fac
        return [rmin, rmax]
    
    def conversion(params, unit):
        params['Rin'] *= unit
        params['Rout'] *= unit
        return params

class powerlaw_errf:
    def val(r, alpha, Rin, Rout, sigma_in=1e-10, sigma_out=1e-10):
        inner_edge = 1-erf((Rin-r)/(np.sqrt(2)*sigma_in*Rin))
        outer_edge = 1-erf((r-Rout)/(np.sqrt(2)*sigma_out*Rout))
        return inner_edge * outer_edge * (r/Rin)**(-alpha)

    def limits(alpha, Rin, Rout, sigma_in=1e-10, sigma_out=1e-10):
        #fac = 3.451 *1.496e13 # 99.99 percentile
        #rmin = max(0, Rin - lin*fac)
        #rmax = Rout + lout*fac
        return [0., Rout + 100 * 1.496e13]
    
    def conversion(params, unit):
        params['Rin'] *= unit
        params['Rout'] *= unit
        return params

class double_powerlaw:
    def val(r, rc, alpha_in, alpha_out, gamma, 
            Rin=None, Rout=None, lin=0.01, lout=0.01):
        value = ((r/rc)**(-alpha_in*gamma) + \
                (r/rc)**(-alpha_out*gamma))**(-1/gamma)
        if Rin:
            value *= (1+np.tanh((r-Rin)/lin))
        if Rout:
            value *= (1+np.tanh((Rout-r)/lout))

        return value / np.max(value)
    
    def limits(rc, alpha_in, alpha_out, gamma,
               Rin=None, Rout=None, lin=0.01, lout=0.01):
        assert alpha_in > 0, "alpha_in must be positive to find inner edge"
        assert alpha_out < 0, "alpha_out must be negative to find outer edge"
        rmax = rc + 5.28e15 + 2.94*(alpha_out**2) + 58.5*alpha_out
        
        fac = 3.451 # 99.9 percentile
        if Rin:
            rmin = max(0, Rin - lin*fac)
        else:
            rmin = 0

        if Rout:
            rmax = min(rmax, Rout + lout*fac)
        return [0., rmax]
    
    def conversion(params, unit):
        params['rc'] *= unit
        if params.get('Rin'):
            params['Rin'] *= unit
        if params.get('Rout'):
            params['Rout'] *= unit
        return params

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
        rmax = Rout + 5.28e15 + 2.94*(alpha_out**2) + 58.5*alpha_out
        return [min(rmin, 0), rmax]
    
    def conversion(params, unit):
        params['Rin'] *= unit
        params['Rout'] *= unit
        return params

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
    def val(x, sigma, R=0):
        return np.exp(-((x-R)**2/(2*sigma**2)))
    
    def norm(sigma, R=0):
        return sigma * np.sqrt(2*np.pi)
    
    def limits(sigma, R=0):
        fac = 2.628 # 99.9 percentile
        rmin = max(0, R - sigma * fac)
        rmax = R + sigma * fac
        return [rmin, rmax]
    
    def conversion(params, unit):
        params['R'] *= unit
        params['sigma'] *= unit
        return params

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
