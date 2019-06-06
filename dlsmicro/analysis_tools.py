import numpy as np


def calc_g1(g2, g0, ergodic, Ip=None, Ie=None, eps=None):
    """Calculate g1 intermediate scattering function from intensity autocorrelation

    Parameters
    __________
    g2 : 1d-array
         Scattering intensity autocorrelation function
    g0 : float
         Intercept of ``g2 - 1`` at time 0
    ergodic: boolean
             If ``False``, corrections are made for non-ergodic systems based on ``Ip`` and ``Ie``
    Ip: float
        Scattering intensity at the position where correlation function is collected.
        (Only used if Ergodic is ``False``)
    Ie: 1d-array
        Vector of scattering intensities at different positions in the cuvette

    Returns
    -------
    g1: 1d-array 
        The intermediate scattering function
    """
    if ergodic:
        g1 = np.sqrt((g2-1.)/g0)
    else:
        Ie_avg = np.average(Ie)
        # calculate the ratio of ensemble to time averaged
        # intensities
        Y = Ie_avg/Ip
        # calculate the g1 correlation function
        if eps is None:
            g1 = (Y-1.)/Y+np.sqrt(g2-g0)/Y
        else:
            g1 = (1.-(1.-eps)/Y +
                  (1-eps)*np.sqrt(1. +
                                  (g2-g0-1.)/(1-eps)**2.)/Y)
    return g1
