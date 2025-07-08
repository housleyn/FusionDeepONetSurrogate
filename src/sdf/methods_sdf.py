import numpy as np

class MethodsSDF:

    def sphere_formation_sdf(self, coords_df, params_df):
        fixed_sphere_coords = [0,0,0]
        fixed_sphere_radius = 1.0
        variable_sphere_radius = 1.0
        variable_sphere_coords = np.column_stack([params_df, np.zeros(params_df.shape[0])])
        coords = coords_df

        distance_fixed = np.linalg.norm(coords - fixed_sphere_coords, axis=1) - fixed_sphere_radius
        distance_variable = np.linalg.norm(coords - variable_sphere_coords, axis=1) - variable_sphere_radius

        sdf = distance_fixed * distance_variable #this could be changed, there are other ways to combine
        return sdf[:, np.newaxis]  # Ensure output is a 2D array with one column