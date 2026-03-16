import numpy as np
from scipy.sparse import coo_matrix

def b14_eq_vert_nodal_force(Beam, Calc):
    """
    Distribute the forces to the degrees of freedom.

    For each time step and depending on the location of the force, the actual
    forces must be distributed to the degrees of freedom; this can be
    accomplished through the shape functions.
    NOTE: Only Vertical forces calculated
    """

    # Initialize variables using Python lists for fast accumulation
    rows = []
    cols = []
    F_vals = []

    num_t = int(Calc.Solver.num_t)

    # Vehicle loop
    for v in range(len(Calc.Veh)):
        
        # Number of loads loop
        num_wheels = Calc.Veh[v].x_path.shape[0]
        
        for wheel in range(num_wheels):
            
            # Time loop
            for t in range(num_t):
                
                elex = int(Calc.Veh[v].elexj[wheel, t])
                
                # In Python, valid elements start at 0. Out-of-bounds were flagged as -1.
                if elex >= 0:
                    
                    x = Calc.Veh[v].xj[wheel, t]
                    a = Beam.Mesh.Ele.a[elex]

                    # DOFs for the element    
                    ele_DOF = Beam.Mesh.Ele.DOF[elex, :]

                    # Append to coordinate lists
                    rows.extend(ele_DOF)
                    cols.extend([t] * 4)  # Repeats the time step index 4 times
                    
                    # Evaluate shape function and multiply by force
                    # Flatten ensures it behaves as a 1D iterable for .extend()
                    shape_vals = Beam.Mesh.shape_fun(x, a).flatten()
                    force_val = Calc.Veh[v].F_onBeam[wheel, t]
                    
                    F_vals.extend(force_val * shape_vals)

    # Create the sparse matrix
    num_dofs = int(Beam.Mesh.DOF.Tnum)
    Fextnew = coo_matrix((F_vals, (rows, cols)), shape=(num_dofs, num_t))

    # Application of boundary conditions to force vector
    # Convert to LIL format briefly to allow row modification, then to CSR for output
    Fextnew = Fextnew.tolil()
    Fextnew[Beam.BC.DOF_fixed, :] = 0.0
    
    return Fextnew.tocsr()

# ---- End of function ----