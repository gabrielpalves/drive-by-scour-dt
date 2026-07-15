"""
Where do high bending moments actually occur on OUR bridges?
Continuous Euler-Bernoulli beam, moving unit load, |M| envelope.
EI cancels out of the envelope SHAPE, so EI=1 is fine.
Supports treated as rigid (our k_v=3.44e8 N/m is stiff); this shifts magnitudes
slightly but not where the peaks are.
"""
import numpy as np

def envelope(spans, nel_per_span=60):
    suppx = np.concatenate([[0.0], np.cumsum(spans)])
    L = suppx[-1]
    # mesh
    nodes = []
    for i, s in enumerate(spans):
        nodes.append(np.linspace(suppx[i], suppx[i+1], nel_per_span+1)[:-1])
    nodes.append([L])
    x = np.unique(np.concatenate(nodes))
    nn = len(x); ndof = 2*nn
    K = np.zeros((ndof, ndof))
    for e in range(nn-1):
        le = x[e+1]-x[e]
        k = (1/le**3)*np.array([
            [12,     6*le,   -12,     6*le],
            [6*le,   4*le**2,-6*le,   2*le**2],
            [-12,   -6*le,    12,    -6*le],
            [6*le,   2*le**2,-6*le,   4*le**2]])
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(d, d)] += k
    # rigid vertical supports
    fixed = [2*int(np.argmin(np.abs(x-sx))) for sx in suppx]
    free = np.setdiff1d(np.arange(ndof), fixed)
    Kff = K[np.ix_(free, free)]
    Kff_inv = np.linalg.inv(Kff)

    Menv = np.zeros(nn-1)
    # move a unit load across every node
    for ln in range(nn):
        F = np.zeros(ndof); F[2*ln] = -1.0
        u = np.zeros(ndof)
        u[free] = Kff_inv @ F[free]
        for e in range(nn-1):
            le = x[e+1]-x[e]
            ue = u[[2*e, 2*e+1, 2*e+2, 2*e+3]]
            # curvature at element midpoint -> M = EI*w''
            xi = 0.5
            B = np.array([(12*xi-6)/le**2, (6*xi-4)/le, (-12*xi+6)/le**2, (6*xi-2)/le])
            M = abs(B @ ue)
            Menv[e] = max(Menv[e], M)
    xm = 0.5*(x[:-1]+x[1:])
    return xm, Menv, suppx

for name, spans in [("L60 / 3-span  (effective 20.1/19.8/20.1)", [20.1, 19.8, 20.1]),
                    ("L99.6 / 4-span (4 x 24.9)",                [24.9]*4)]:
    xm, Me, suppx = envelope(spans)
    Me_n = Me/Me.max()
    L = suppx[-1]
    print(f"\n=== {name} ===")
    print(f"supports at: {np.round(suppx,2)}")
    # local maxima of the envelope
    peaks = [i for i in range(1, len(Me)-1) if Me[i] >= Me[i-1] and Me[i] >= Me[i+1]]
    print("envelope peaks (crack-prone sections):")
    for i in peaks:
        near = min(suppx, key=lambda s: abs(s-xm[i]))
        kind = "OVER-PIER (hogging)" if abs(near-xm[i]) < 1.0 and 0 < near < L else "mid-span (sagging)"
        print(f"   x = {xm[i]:6.2f} m  (x/L = {xm[i]/L:.3f})   |M|/|M|max = {Me_n[i]:.2f}   {kind}")
    # where is the envelope LOW? (cracks unlikely there)
    lo = xm[(Me_n < 0.35) & (xm > 0.10*L) & (xm < 0.90*L)]
    if len(lo):
        # group contiguous
        grp, cur = [], [lo[0]]
        for a, b in zip(lo[:-1], lo[1:]):
            if b-a < 1.0: cur.append(b)
            else: grp.append(cur); cur = [b]
        grp.append(cur)
        print("LOW-moment zones inside the current uniform crack range [0.10L, 0.90L]")
        print("  (|M| < 35% of max -> a crack here is both unlikely AND a weak nuisance):")
        for g in grp:
            print(f"   x = {g[0]:6.2f} - {g[-1]:6.2f} m   (x/L = {g[0]/L:.3f} - {g[-1]/L:.3f})")
    frac_low = float(((Me_n < 0.35) & (xm > 0.10*L) & (xm < 0.90*L)).sum()) / \
               float(((xm > 0.10*L) & (xm < 0.90*L)).sum())
    print(f"=> {frac_low*100:.0f}% of the current uniform [0.10L, 0.90L] draw lands in low-moment zones")
