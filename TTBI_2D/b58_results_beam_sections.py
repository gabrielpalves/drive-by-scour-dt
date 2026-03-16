import numpy as np
from scipy.interpolate import interp1d

def b58_results_beam_sections(Sol, Beam, Calc):
    """
    Calculates the available Beam results at predefined sections.
    """

    # Check if we have sections to calculate
    if getattr(Calc.Options, 'num_calc_beam_sections', 0) > 0:
        
        # Ensure sections is an iterable array
        sections = np.atleast_1d(Calc.Options.calc_beam_sections)
        x_nodes = Beam.Mesh.Nodes.acum

        # Loop through all dynamically created result fields in Sol.Beam (U, BM, Shear, Acc, etc.)
        # vars() accesses the object's dictionary of attributes
        for field_name, field_obj in vars(Sol.Beam).items():
            
            # Only attempt interpolation if the field contains the 'xt' time-history matrix
            if hasattr(field_obj, 'xt'):
                
                # Create a vectorized 1D interpolator along the spatial axis (axis=0).
                # bounds_error=False and fill_value=np.nan perfectly replicate MATLAB's 
                # default behavior of returning NaN if a section falls outside the beam.
                f_interp = interp1d(
                    x_nodes, 
                    field_obj.xt, 
                    axis=0, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=np.nan
                )
                
                # Evaluate the interpolator for all sections at once. 
                # This directly yields a (num_sections x num_t) array.
                field_obj.sections_t = f_interp(sections)

    return Sol

# ---- End of function ----