import numpy as np

class MethodsSDF:

    def sphere_formation_sdf(self, coords_df, params_df):

        # Convert to numpy arrays
        coords = np.asarray(coords_df, dtype=float)  # (N, 3)
        params = np.asarray(params_df, dtype=float)  # shape (N, 2), same param repeated

        # Ensure coords are 3D
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(coords.shape[0])])  # add z=0

        # Extract second sphere center from first row of params (assumes it's repeated)
        center2 = np.append(params[0], 0.0)  # (x, y, 0)
        
        # Fixed sphere at origin
        center1 = np.array([0.0, 0.0, 0.0])
        
        # Radii 
        r1 = 1.0
        r2 = 1.0

        # Compute signed distances
        sdf1 = np.linalg.norm(coords - center1, axis=1) - r1
        sdf2 = np.linalg.norm(coords - center2, axis=1) - r2

        # Stack both SDFs: shape (N, 2)
        sdf = np.stack([sdf1, sdf2], axis=1)
        
        return sdf