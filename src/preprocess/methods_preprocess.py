import pandas as pd
import numpy as np
from scipy.stats import qmc 
from sklearn.neighbors import NearestNeighbors

class MethodsPreprocess:
    def load_and_pad(self):
        
        for path in self.files:
            df = pd.read_csv(path)
            coords_full = df[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
            outputs_full = df[["Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)", "Absolute Pressure (Pa)"]].to_numpy()
            if self.dimension == 3:
                coords, outputs = self.LHS(coords_full, outputs_full)
                self.lhs_applied = True
            else:
                coords = coords_full
                outputs = outputs_full

            
            param_vec = df[self.param_columns].to_numpy()
            # param_vec = self._reduce_params(param_vec)

            self.coords.append(coords)
            self.params.append(param_vec)
            self.outputs.append(outputs)

        self.npts_max = max(c.shape[0] for c in self.coords)
        
        if self.dimension ==2:
            
            padded_coords = []
            padded_params = []
            padded_outputs = []
            for c, r, o in zip(self.coords, self.params, self.outputs):
                c_pad = self._pad(c)
                r_pad = self._pad(r)
                o_pad = self._pad(o)
                padded_coords.append(c_pad)
                padded_params.append(r_pad)
                padded_outputs.append(o_pad)

            self.coords = padded_coords
            self.params = padded_params
            self.outputs = padded_outputs
        
        outputs_flat = np.vstack(self.outputs)
        outputs_flat, self.outputs_mean, self.outputs_std = self._normalize(outputs_flat)

        idx = 0
        for i in range(len(self.outputs)):
            n = self.outputs[i].shape[0]
            self.outputs[i] = outputs_flat[idx:idx+n]
            idx += n

    def LHS(self, coords_full, outputs_full): 
        N_sample = self.lhs_sample
        coords_min = coords_full.min(axis=0)
        coords_max = coords_full.max(axis=0)
        coords_norm = (coords_full - coords_min) / (coords_max - coords_min)

        
        sampler = qmc.LatinHypercube(d=3)
        lhs = sampler.random(n=N_sample)

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords_norm)
        _, indices = nn.kneighbors(lhs)
        indices = indices[:, 0]

        coords = coords_full[indices]
        outputs = outputs_full[indices]
        return coords, outputs

    def _reduce_params(self, param_vec):
        """Reduce parameter dimensionality.

        Currently if two columns are provided, compute the aspect ratio
        by dividing the first column by the second.
        """
        param_vec = np.asarray(param_vec)
        if param_vec.ndim == 2 and param_vec.shape[1] == 2:
            aspect_ratio = param_vec[:, 0] / param_vec[:, 1]
            return aspect_ratio[:, None]
        return param_vec

    def _pad(self, arr):
        arr = np.asarray(arr)
        n_pad = self.npts_max - arr.shape[0]
        padded = np.pad(arr, ((0, n_pad), (0, 0)), mode="edge")
        return padded

    def to_numpy(self):
        X_coords = np.stack(self.coords)
        Y_outputs = np.stack(self.outputs)
        G_params = np.stack(self.params)[:, 0, :]
        return X_coords, Y_outputs, G_params

    def save(self):
        X_coords, Y_outputs, G_params = self.to_numpy()
        np.savez(
            self.output_path,
            coords=X_coords,           # shape: (num_samples, npts_max, 3)
            outputs=Y_outputs,         # shape: (num_samples, npts_max, output_dim)
            params=G_params,           # shape: (num_samples, param_dim)
            outputs_mean=self.outputs_mean,   # shape: (output_dim,)
            outputs_std=self.outputs_std,     # shape: (output_dim,)
        )

    def run_all(self):
        self.load_and_pad()
        self.save()

    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  
        return (data - mean) / std, mean, std