import pandas as pd
import numpy as np
from scipy.stats import qmc 
from sklearn.neighbors import NearestNeighbors



class Preprocess:
    def __init__(self, radius_files, output_path="processed_data.npz"):
        self.radius_files = radius_files
        self.output_path = output_path
        self.coords = []
        self.radii = []
        self.outputs = []
        self.npts_max = 0  # max elements determined based on data

    def load_and_pad(self):
        for radius, path in self.radius_files.items():
            df = pd.read_csv(path)
            coords_full = df[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
            outputs_full = df[["Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)", "Absolute Pressure (Pa)"]].to_numpy()

            N_sample = 50000
            coords_min = coords_full.min(axis=0)
            coords_max = coords_full.max(axis=0)
            coords_norm = (coords_full - coords_min) / (coords_max - coords_min)

            # LHS points in [0, 1]^3
            sampler = qmc.LatinHypercube(d=3)
            lhs = sampler.random(n=N_sample)

            # Nearest neighbor mapping
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords_norm)
            _, indices = nn.kneighbors(lhs)
            indices = indices[:, 0]

            # Select matched points
            coords = coords_full[indices]
            outputs = outputs_full[indices]


            radius_vec = np.full((coords.shape[0], 1), radius)

            self.coords.append(coords)
            self.radii.append(radius_vec)
            self.outputs.append(outputs)

        self.npts_max = max(c.shape[0] for c in self.coords)

        self.coords = [self._pad(c) for c in self.coords]
        self.radii = [self._pad(r) for r in self.radii]
        self.outputs = [self._pad(o) for o in self.outputs]

        # Normalize AFTER padding
        coords_flat = np.vstack(self.coords)
        outputs_flat = np.vstack(self.outputs)
        radii_flat = np.vstack(self.radii)

        coords_flat, self.coords_mean, self.coords_std = self._normalize(coords_flat)
        outputs_flat, self.outputs_mean, self.outputs_std = self._normalize(outputs_flat)
        radii_flat, self.radii_mean, self.radii_std = self._normalize(radii_flat)

        # Reassign normalized values
        idx = 0
        for i in range(len(self.coords)):
            n = self.coords[i].shape[0]
            self.coords[i] = coords_flat[idx:idx+n]
            self.outputs[i] = outputs_flat[idx:idx+n]
            self.radii[i] = radii_flat[idx:idx+n]
            idx += n

    def _pad(self, arr):
        return np.pad(arr, ((0, self.npts_max - arr.shape[0]), (0, 0)), mode="edge")

    def to_numpy(self):
        X_coords = np.stack(self.coords)
        Y_outputs = np.stack(self.outputs)
        G_params = np.stack(self.radii)[:, 0, :]  # extract 1 value per sample
        return X_coords, Y_outputs, G_params

    def save(self):
        X_coords, Y_outputs, G_params = self.to_numpy()
        np.savez(self.output_path, coords=X_coords, outputs=Y_outputs, params=G_params,
                coords_mean=self.coords_mean,
                coords_std=self.coords_std,
                outputs_mean=self.outputs_mean,
                outputs_std=self.outputs_std,
                radii_mean=self.radii_mean,
                radii_std=self.radii_std)

    def run_all(self):
        self.load_and_pad()
        self.save()

    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # avoid division by zero
        return (data - mean) / std, mean, std
