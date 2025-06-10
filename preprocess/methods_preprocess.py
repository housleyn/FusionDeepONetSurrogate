import pandas as pd
import numpy as np
from scipy.stats import qmc 
from sklearn.neighbors import NearestNeighbors

class MethodsPreprocess:
    def load_and_pad(self):
        for radius, path in self.radius_files.items():
            df = pd.read_csv(path)
            coords_full = df[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
            outputs_full = df[["Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)", "Absolute Pressure (Pa)"]].to_numpy()
            if self.dimension == 3:
                coords, outputs = self.LHS(coords_full, outputs_full)
                self.lhs_applied = True
            else:
                coords = coords_full
                outputs = outputs_full
            
            radius_vec = np.full((coords.shape[0], 1), radius)

            self.coords.append(coords)
            self.radii.append(radius_vec)
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

        # Normalize AFTER padding
        coords_flat = np.vstack(self.coords)
        outputs_flat = np.vstack(self.outputs)
        radii_flat = np.vstack(self.radii)

        coords_flat, self.coords_mean, self.coords_std = self._normalize(coords_flat)
        outputs_flat, self.outputs_mean, self.outputs_std = self._normalize(outputs_flat)
        radii_flat, self.radii_mean, self.radii_std = self._normalize(radii_flat)

        # unflatten after the flattening for normalization, 
        idx = 0
        for i in range(len(self.coords)):
            n = self.coords[i].shape[0]
            self.coords[i] = coords_flat[idx:idx+n]
            self.outputs[i] = outputs_flat[idx:idx+n]
            self.radii[i] = radii_flat[idx:idx+n]
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
        G_params = np.stack(self.radii)[:, 0, :]  # extract 1 value per sample
        return X_coords, Y_outputs, G_params

    def save(self):
        X_coords, Y_outputs, G_params = self.to_numpy()
        np.savez(
            self.output_path,
            coords=X_coords,           # shape: (num_samples, npts_max, 3)
            outputs=Y_outputs,         # shape: (num_samples, npts_max, output_dim)
            params=G_params,           # shape: (num_samples, 1)
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