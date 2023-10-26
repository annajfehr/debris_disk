import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve 

class powerlaw:
    def val(r, alpha, Rin, Rout, lin=0.01, lout=0.01):
        return (r/Rin)**alpha * (1+erf((r-Rin)/lin)) * \
                (1+erf((Rout-r)/lout))/4
    
    def limits(alpha, Rin, Rout, lin=0.01, lout=0.01):
        fac = 3.451 *1.496e13 # 99.99 percentile
        rmin = max(0, Rin - lin*fac)
        rmax = Rout + lout*fac
        return [rmin, rmax]
    
    def conversion(params, unit):
        params['Rin'] *= unit
        params['Rout'] *= unit
        if params.get('lin'):
            params['lin'] *= unit
        if params.get('lout'):
            params['lout'] *= unit
        return params

class powerlaw_errf:
    def val(r, alpha, Rin, Rout, sigma_in=1e-10, sigma_out=1e-10):
        inner_edge = 1-erf((Rin-r)/(np.sqrt(2)*sigma_in*Rin))
        outer_edge = 1-erf((r-Rout)/(np.sqrt(2)*sigma_out*Rout))
        return inner_edge * outer_edge * (r/Rin)**(-alpha)

    def limits(alpha, Rin, Rout, sigma_in=1e-10, sigma_out=1e-10):
        max_val = Rin * powerlaw_errf.val(Rin, alpha, Rin, Rout, sigma_in, sigma_out)
        f = lambda x : (x * powerlaw_errf.val(x, alpha, Rin, Rout, sigma_in, sigma_out)) - max_val * 0.01
        Rmax = fsolve(f, Rout)[0]
        if Rmax < Rout:
            Rmax = 2 * Rout
        return [0., Rmax]
    
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
            value *= (1+erf((r-Rin)/lin))
        if Rout:
            value *= (1+erf((Rout-r)/lout))

        return value
    
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

        max_val = rc * double_powerlaw.val(rc, rc, alpha_in, alpha_out, gamma, Rin, Rout, lin, lout)
        f = lambda x : (x * double_powerlaw.val(x, rc, alpha_in, alpha_out, gamma, Rin, Rout, lin, lout)) - max_val * 0.1
        Rmax = fsolve(f, rc*3)[0]
        if Rmax < rc:
            Rmax = rmax
        if Rmax < rmin:
            Rmax = 3.0*np.abs(rc)
            rmin = 0
        return [rmin, Rmax]
    
    def conversion(params, unit):
        params['rc'] *= unit
        if params.get('Rin'):
            params['Rin'] *= unit
        if params.get('lin'):
            params['lin'] *= unit
        if params.get('Rout'):
            params['Rout'] *= unit
        if params.get('lout'):
            params['lout'] *= unit
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

class single_erf:
    def val(r, Rc, sigma_in, alpha_out):
        inner_edge = 1-erf((Rc-r)/(np.sqrt(2)*sigma_in*Rc))
        return inner_edge * (r/Rc)**-alpha_out

    def limits(Rc, sigma_in, alpha_out):
        return [0, np.inf] #filler, will use [resolution/2, Rmax)

    def conversion(params, unit):
        params['Rc'] *= unit
        return params

class asymmetric_gaussian:
    def val(r, Rc, sigma_in, sigma_out):
        return np.piecewise(r, [r < Rc, r >= Rc], [lambda r : gaussian.val(r, sigma_in, Rc), lambda r : gaussian.val(r, sigma_out, Rc)])

    def limits(Rc, sigma_in, sigma_out):
        return [0, np.inf] #filler, will use [resolution/2, Rmax)
    
    def conversion(params, unit):
        params['Rc'] *= unit
        params['sigma_in'] *= unit
        params['sigma_out'] *= unit
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
    def val(x, sigma, R=0, gamma=2):
        return np.exp(-((x-R)**gamma/(2*sigma**gamma)))
    
    def norm(sigma, R=0, gamma=2):
        return sigma * np.sqrt(2*np.pi)
    
    def limits(sigma, R=0, gamma=2):
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
