import numpy as np

class MethodsSDF:

    def sphere_formation_sdf(self, coords_df, params_df):
        """Compute signed distance to a sphere.

        Parameters
        ----------
        coords_df : array_like, shape (N, 3)
            Cartesian coordinates of the points.
        params_df : array_like, shape (N, P)
            Parameters describing the sphere. Accepted forms are:
            - ``P == 1``: sphere radius, centre assumed at the origin.
            - ``P == 2``: x and y coordinates of the centre, z is assumed 0 and
              radius is 1.
            - ``P >= 4``: first three columns are centre coordinates, fourth is
              the radius.
        Returns
        -------
        np.ndarray, shape (N, 1)
            Signed distance of each point to the sphere surface.
        """

        coords = np.asarray(coords_df, dtype=float)
        params = np.asarray(params_df, dtype=float)

        if params.ndim == 1:
            params = params[:, None]

        if params.shape[1] == 1:
            centers = np.zeros((coords.shape[0], 3))
            radii = params[:, 0]
        elif params.shape[1] == 2:
            centers = np.column_stack([params, np.zeros(params.shape[0])])
            radii = np.ones(params.shape[0])
        elif params.shape[1] >= 4:
            centers = params[:, :3]
            radii = params[:, 3]
        else:
            raise ValueError("params_df has an unsupported shape")

        distance = np.linalg.norm(coords - centers, axis=1) - radii

        return distance[:, np.newaxis]