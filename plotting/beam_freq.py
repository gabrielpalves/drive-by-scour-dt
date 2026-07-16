"""Natural frequencies of OUR deck with rho=9.6 (current code) vs rho=9600
(Fernandes, per the user). Euler-Bernoulli FE, consistent mass, vertical support
springs k_v0 = 344e6 N/m (B02_BoundaryConditions.m:65). Deck alone (the track
layer adds mass and would lower these slightly)."""
import numpy as np
from scipy.linalg import eigh

E, I, A = 3.5e10, 0.33, 1.0          # A03 / B43
KV0 = 344e6                          # B02 healthy vertical support stiffness [N/m]

def freqs(spans, rho, nel_per_span=40, n=3):
    suppx = np.concatenate([[0.0], np.cumsum(spans)])
    x = np.unique(np.concatenate([np.linspace(suppx[i], suppx[i+1], nel_per_span+1)
                                  for i in range(len(spans))]))
    nn = len(x); nd = 2*nn
    K = np.zeros((nd, nd)); M = np.zeros((nd, nd))
    for e in range(nn-1):
        le = x[e+1]-x[e]
        ke = (E*I/le**3)*np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]])
        me = (rho*A*le/420)*np.array([
            [156, 22*le, 54, -13*le],
            [22*le, 4*le**2, 13*le, -3*le**2],
            [54, 13*le, 156, -22*le],
            [-13*le, -3*le**2, -22*le, 4*le**2]])
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(d, d)] += ke; M[np.ix_(d, d)] += me
    for sx in suppx:                       # vertical support springs
        K[2*int(np.argmin(np.abs(x-sx))), 2*int(np.argmin(np.abs(x-sx)))] += KV0
    w2 = eigh(K, M, eigvals_only=True)
    w2 = w2[w2 > 1e-6]
    return np.sqrt(w2[:n])/(2*np.pi)

print(f"EI = {E*I:.3e} N.m^2 ;  k_v0 = {KV0:.3g} N/m\n")
for name, spans in [("Fernandes / champion  L40 = 2 x 20 m", [20.0, 20.0]),
                    ("ladder L60 = 3 spans (eff. 20.1/19.8/20.1)", [20.1, 19.8, 20.1]),
                    ("ladder L99.6 = 4 x 24.9 m", [24.9]*4)]:
    f_bug = freqs(spans, 9.6)
    f_fix = freqs(spans, 9600.0)
    print(f"{name}")
    print(f"   rho =    9.6  (CURRENT CODE) -> f1..f3 = {np.round(f_bug,1)} Hz")
    print(f"   rho = 9600.0  (Fernandes)    -> f1..f3 = {np.round(f_fix,2)} Hz")
    print(f"   ratio f1: {f_bug[0]/f_fix[0]:.1f}x   (sqrt(9600/9.6) = {np.sqrt(1000):.1f})\n")

print("Real railway bridge spans of ~20 m sit at ~3-6 Hz.")
