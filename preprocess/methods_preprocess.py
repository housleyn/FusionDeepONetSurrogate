import pandas as pd
import numpy as np
from scipy.stats import qmc 
from sklearn.neighbors import NearestNeighbors

class MethodsPreprocess:
    def load_and_pad(self):
        # ``self.files`` maps a parameter (radius or ``(a,b)`` tuple) to a CSV path
        for param, path in self.files.items():
            df = pd.read_csv(path)
            coords_full = df[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
            outputs_full = df[["Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)", "Absolute Pressure (Pa)"]].to_numpy()
            if self.dimension == 3:
                coords, outputs = self.LHS(coords_full, outputs_full)
                self.lhs_applied = True
            else:
                coords = coords_full
                outputs = outputs_full

            if {"a", "b"}.issubset(df.columns):
                param_vec = df[["a", "b"]].to_numpy()
            else:
                param_vec = np.full((coords.shape[0], 1), float(param))

            self.coords.append(coords)
            self.radii.append(param_vec)
            self.outputs.append(outputs)

        self.npts_max = max(c.shape[0] for c in self.coords)

        # pad all arrays to the maximum number of points
        padded_coords = []
        padded_radii = []
        padded_outputs = []
        for c, r, o in zip(self.coords, self.radii, self.outputs):
            c_pad = self._pad(c)
            r_pad = self._pad(r)
            o_pad = self._pad(o)
            padded_coords.append(c_pad)
            padded_radii.append(r_pad)
            padded_outputs.append(o_pad)

        self.coords = padded_coords
        self.radii = padded_radii
        self.outputs = padded_outputs

        # Compute statistics without normalizing coordinates or radii
        coords_flat = np.vstack(self.coords)
        outputs_flat = np.vstack(self.outputs)
        radii_flat = np.vstack(self.radii)

        # store stats for reference but only normalize flow variables
        self.coords_mean = np.mean(coords_flat, axis=0)
        self.coords_std = np.std(coords_flat, axis=0)
        self.coords_std[self.coords_std == 0] = 1

        self.radii_mean = np.mean(radii_flat, axis=0)
        self.radii_std = np.std(radii_flat, axis=0)
        self.radii_std[self.radii_std == 0] = 1

        outputs_flat, self.outputs_mean, self.outputs_std = self._normalize(outputs_flat)

        # replace outputs with normalized values
        idx = 0
        for i in range(len(self.outputs)):
            n = self.outputs[i].shape[0]
            self.outputs[i] = outputs_flat[idx:idx+n]
            idx += n

    def LHS(self, coords_full, outputs_full): # Latin Hypercube Sampling, 500,000 samples per simulation
        N_sample = 500000
        coords_min = coords_full.min(axis=0)
        coords_max = coords_full.max(axis=0)
        coords_norm = (coords_full - coords_min) / (coords_max - coords_min)

        
        sampler = qmc.LatinHypercube(d=3)
        lhs = sampler.random(n=N_sample)

        # Nearest neighbor mapping
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords_norm)
        _, indices = nn.kneighbors(lhs)
        indices = indices[:, 0]

        # Select matched points
        coords = coords_full[indices]
        outputs = outputs_full[indices]
        return coords, outputs
    
    def _pad(self, arr):
        arr = np.asarray(arr)
        n_pad = self.npts_max - arr.shape[0]
        padded = np.pad(arr, ((0, n_pad), (0, 0)), mode="edge")
        return padded

    def to_numpy(self):
        X_coords = np.stack(self.coords)
        Y_outputs = np.stack(self.outputs)
        G_params = np.stack(self.radii)[:, 0, :]
        return X_coords, Y_outputs, G_params

    def save(self):
        X_coords, Y_outputs, G_params = self.to_numpy()
        np.savez(
            self.output_path,
            coords=X_coords,           # shape: (num_samples, npts_max, 3)
            outputs=Y_outputs,         # shape: (num_samples, npts_max, output_dim)
            params=G_params,           # shape: (num_samples, 2)
            coords_mean=self.coords_mean,     # shape: (3,)
            coords_std=self.coords_std,       # shape: (3,)
            outputs_mean=self.outputs_mean,   # shape: (output_dim,)
            outputs_std=self.outputs_std,     # shape: (output_dim,)
            radii_mean=self.radii_mean,       # shape: (1,)
            radii_std=self.radii_std          # shape: (1,)
        )

    def run_all(self):
        self.load_and_pad()
        self.save()

    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # avoid division by zero
        return (data - mean) / std, mean, std

