import numpy as np
import matplotlib.pyplot as plt

def b50_element_num_of_force(Beam, Calc):
    """
    Calculates the element and relative position of each wheel for each vehicle in time.
    """

    # Vehicle loop
    for v in range(len(Calc.Veh)):
        
        # x_path shape: (num_wheels, num_time_steps)
        x_path = Calc.Veh[v].x_path
        
        # Vectorized Determination of element number
        # searchsorted efficiently finds the index where x_path would fit in the Nodes array.
        # Subtracting 1 inherently provides a 0-based element index for Python.
        elexj = np.searchsorted(Beam.Mesh.Nodes.acum, x_path, side='right') - 1
        
        # Clip indices to prevent out-of-bounds errors for positions past the last node
        elexj_clipped = np.clip(elexj, 0, Beam.Mesh.Ele.Tnum - 1)
        
        # Calculate relative position within the element (xj)
        xj = x_path - Beam.Mesh.Nodes.acum[elexj_clipped]
        
        # Handle positions before the start of the beam (x < 0)
        # In MATLAB, 0 was assigned. Since Python is 0-indexed, element 0 is valid.
        # We assign -1 to explicitly flag "Out of bounds / No element"
        before_start = x_path < Beam.Mesh.Nodes.acum[0]
        elexj_clipped[before_start] = -1
        xj[before_start] = 0.0
        
        Calc.Veh[v].elexj = elexj_clipped
        Calc.Veh[v].xj = xj

        # Redux option boundaries
        if getattr(Calc.Options, 'redux', 0) == 1:
            out_of_bounds = (x_path < 0) | (x_path > Calc.Profile.L)
            # Flagging out-of-bounds as -1 to safely ignore them during force assignment later
            Calc.Veh[v].elexj[out_of_bounds] = -1
            Calc.Veh[v].xj[out_of_bounds] = 0.0
            
        # # Graphical check
        # plt.figure()
        # 
        # plt.subplot(2, 1, 1)
        # plt.plot(Calc.Solver.t, Calc.Veh[v].elexj.T)
        # plt.ylabel('Element num. (0-based)')
        # 
        # plt.subplot(2, 1, 2)
        # plt.plot(Calc.Solver.t, Calc.Veh[v].xj.T)
        # plt.ylabel('x in element')
        # 
        # plt.tight_layout()
        # plt.show()

    return Calc

# ---- End of function ----