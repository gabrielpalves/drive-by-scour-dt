import numpy as np

# A lightweight dummy class to serve as a structure for missing nested fields
class EmptyObj:
    pass

def b01_elements_and_coordinates(Beam, Calc=None):
    """
    Calculation of node coordinates and associated DOF to each element.
    Specifies the properties element by element for (E, I, rho and A).
    Also generates additional and auxiliary variables needed in the FEM model.
    """

    # Mesh definition 
    num_ele = Beam.Mesh.Ele.num
    L = Beam.Prop.L
    
    Beam.Mesh.Ele.a = np.ones(num_ele) * (L / num_ele)
    # np.cumsum calculates the cumulative sum. We prepend 0 for the starting coordinate.
    Beam.Mesh.Nodes.acum = np.concatenate(([0.0], np.cumsum(Beam.Mesh.Ele.a)))

    # -- Coordinates for each node (nodes_coord) --
    # Reshape converts the 1D array into a column vector (similar to MATLAB's transpose ')
    Beam.Mesh.Nodes.coord = Beam.Mesh.Nodes.acum.reshape(-1, 1)
    Beam.Mesh.Ele.num_nodes = 2
    Beam.Mesh.Nodes.num_DOF = 2
    Beam.Mesh.Ele.Tnum = num_ele
    
    # 0-based node generation for each element
    # MATLAB: [(1:num); (1:num)+1]'  ->  Python: [[0, 1], [1, 2], [2, 3], ...]
    n1 = np.arange(num_ele)
    n2 = n1 + 1
    Beam.Mesh.Ele.nodes = np.column_stack((n1, n2))
    
    # 0-based DOF generation for each element
    # MATLAB: [DOF, DOF+1, DOF+2, DOF+3] starting from 1
    # Python: [[0, 1, 2, 3], [2, 3, 4, 5], ...] starting from 0
    dof_start = np.arange(0, num_ele * 2, 2).reshape(-1, 1)
    Beam.Mesh.Ele.DOF = np.hstack((dof_start, dof_start + 1, dof_start + 2, dof_start + 3))
    
    Beam.Mesh.Nodes.Tnum = Beam.Mesh.Ele.Tnum + 1
    Beam.Mesh.DOF.Tnum = Beam.Mesh.Nodes.Tnum * Beam.Mesh.Nodes.num_DOF

    # -- Element by element property definition Input processing --
    # Helper lambda to check if a property is a scalar or an array of length 1
    is_scalar_like = lambda x: np.isscalar(x) or (isinstance(x, (list, np.ndarray)) and len(x) == 1)

    # Beam Young's Modulus
    if is_scalar_like(Beam.Prop.E):
        Beam.Prop.E_n = np.ones(Beam.Mesh.Ele.Tnum) * Beam.Prop.E
        
    # Beam's section Second moment of Area
    if is_scalar_like(Beam.Prop.I):
        Beam.Prop.I_n = np.ones(Beam.Mesh.Ele.Tnum) * Beam.Prop.I
        
    # Beam's density
    if is_scalar_like(Beam.Prop.rho):
        Beam.Prop.rho_n = np.ones(Beam.Mesh.Ele.Tnum) * Beam.Prop.rho
        
    # Beam's section Area
    if is_scalar_like(Beam.Prop.A):
        Beam.Prop.A_n = np.ones(Beam.Mesh.Ele.Tnum) * Beam.Prop.A

    # ---- Optional calculations ----
    # Triggered if the optional `Calc` argument is provided
    if Calc is not None:
        # Initialize Mid structure if it doesn't exist
        if not hasattr(Beam.Mesh.Nodes, 'Mid'):
            Beam.Mesh.Nodes.Mid = EmptyObj()
        if not hasattr(Beam.Mesh, 'DOF') or not hasattr(Beam.Mesh.DOF, 'Mid'):
            if not hasattr(Beam.Mesh, 'DOF'): Beam.Mesh.DOF = EmptyObj()
            Beam.Mesh.DOF.Mid = EmptyObj()

        # -- Mid-span information --
        diffs = np.abs(Beam.Mesh.Nodes.acum - (Beam.Prop.L / 2.0))
        min_ind = np.argmin(diffs)
        min_value = diffs[min_ind]
        
        if min_value < Calc.Cte.tol:
            Beam.Mesh.Nodes.Mid.exists = 1
            Beam.Mesh.Nodes.Mid.node = min_ind # This is now a 0-based node index
            
            # Vertical DOF at mid-span. 
            # MATLAB 1-based formula: (num_DOF * node) - 1
            # Python 0-based formula: num_DOF * node
            Beam.Mesh.DOF.Mid.vert = Beam.Mesh.Nodes.num_DOF * Beam.Mesh.Nodes.Mid.node
        else:
            Beam.Mesh.Nodes.Mid.exists = 0

    return Beam

# ---- End of function ----