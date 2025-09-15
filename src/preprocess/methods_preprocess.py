import pandas as pd
import numpy as np
from scipy.stats import qmc 
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

class MethodsPreprocess:
    def load_data(self):
            for path in self.files:
                print(f"Loading data from {path}")
                df = pd.read_csv(path)
                coords_full = df[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
                outputs_full = df[["Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)", "Absolute Pressure (Pa)", "Temperature (K)"]].to_numpy()
                sdf_full = df[self.distance].to_numpy()
                
                if self.dimension == 3:
                    coords, outputs = self.LHS(coords_full, outputs_full, sdf_full)
                    self.lhs_applied = True
                    
                else:
                    # weights = self.get_weights(coords_full[:, 0], coords_full[:, 1], outputs_full[:, 3])
                    coords = coords_full
                    outputs = outputs_full
                    sdf = sdf_full

                
                param_vec = df[self.param_columns].to_numpy()
                if param_vec.shape[0] != coords.shape[0]:
                    param_vec = np.repeat(param_vec[:1], coords.shape[0], axis=0)
                

                self.coords.append(coords)
                self.params.append(param_vec)
                self.outputs.append(outputs)
                self.sdf.append(sdf)
                # self.weights.append(weights)


    def load_and_pad(self):
        
        self.load_data()

        self.npts_max = max(c.shape[0] for c in self.coords)
        
        if self.dimension ==2:
            
            padded_coords = []
            padded_params = []
            padded_outputs = []
            padded_sdf = []
            # padded_weights = []
            for c, r, o, s in zip(self.coords, self.params, self.outputs, self.sdf):
                c_pad = self._pad(c)
                r_pad = self._pad(r)
                o_pad = self._pad(o)
                s_pad = self._pad(s)
                # w_pad = np.pad(w, (0,self.npts_max-w.shape[0]), mode="edge")
                padded_coords.append(c_pad)
                padded_params.append(r_pad)
                padded_outputs.append(o_pad)
                padded_sdf.append(s_pad)
                # padded_weights.append(w_pad)


            self.sdf = padded_sdf
            # self.weights = padded_weights
            self.coords = padded_coords
            self.params = padded_params
            self.outputs = padded_outputs
            
        
        outputs_flat = np.vstack(self.outputs)
        # outputs_flat = self._subtract_free_stream(outputs_flat)
        outputs_flat, self.outputs_mean, self.outputs_std = self._normalize(outputs_flat)

        idx = 0
        for i in range(len(self.outputs)):
            n = self.outputs[i].shape[0]
            self.outputs[i] = outputs_flat[idx:idx+n]
            idx += n

    def LHS(self, coords_full, outputs_full, sdf_full): 
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
        sdf = sdf_full[indices]
        return coords, outputs, sdf

    def compute_gradient_unstructured(self, x, y, f, k=6):
            coords = np.column_stack((x, y))
            tree = KDTree(coords)
            grad = np.zeros_like(f)

            for i, (xi, yi) in enumerate(coords):
                dists, idxs = tree.query([xi, yi], k=k)
                A = np.c_[x[idxs] - xi, y[idxs] - yi]
                b = f[idxs] - f[i]
                g, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                grad[i] = np.linalg.norm(g)
            return grad
    
    def filter_high_gradient(self, grad_p, threshold=10000000):
        high_grad_mask = grad_p > threshold
        return high_grad_mask
    
    def compute_weights(self, high_grad_mask):
        weights = np.ones_like(high_grad_mask, dtype=float)
        weights[high_grad_mask] = 250  
        return weights
    
    def get_weights(self, x, y, absolute_pressure):
        grad_p = self.compute_gradient_unstructured(x, y, absolute_pressure)
        high_grad_mask = self.filter_high_gradient(grad_p)
        weights = self.compute_weights(high_grad_mask)
        return weights

    def _pad(self, arr):
        arr = np.asarray(arr)
        n_pad = self.npts_max - arr.shape[0]
        padded = np.pad(arr, ((0, n_pad), (0, 0)), mode="edge")
        return padded

    def to_numpy(self):
        X_coords = np.stack(self.coords)
        Y_outputs = np.stack(self.outputs)
        G_params = np.stack(self.params)[:, 0, :]
        S_sdf = np.stack(self.sdf)
        return X_coords, Y_outputs, G_params, S_sdf

    def save(self):
        X_coords, Y_outputs, G_params, S_sdf = self.to_numpy()
        np.savez(
            self.output_path,
            coords=X_coords,           # shape: (num_samples, npts_max, 3)
            outputs=Y_outputs,         # shape: (num_samples, npts_max, output_dim)
            params=G_params,           # shape: (num_samples, param_dim)
            sdf=S_sdf,              # shape: (num_samples, npts_max, 1)
            # weights=self.weights,       # shape: (num_samples, npts_max, 1)
            outputs_mean=self.outputs_mean,   # shape: (output_dim,)
            outputs_std=self.outputs_std,     # shape: (output_dim,)

        )

    def run_all(self):
        self.load_and_pad()
        print("saving preprocessed data to", self.output_path)
        self.save()
    def run_all_low_fi(self):
        self.load_and_pad()
        print("saving preprocessed low fidelity data to", self.output_path)
        self.save()

    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  
        return (data - mean) / std, mean, std

    def _subtract_free_stream(self, data):

        self.freestream_by_name = {
            "Density (kg/m^3)":       1.1766728065550904,
            "Velocity[i] (m/s)":      -3472.4454196563174,
            "Velocity[j] (m/s)":      0.0,
            "Velocity[k] (m/s)":      0.0,
            "Absolute Pressure (Pa)": 101324.99849262246,
            "Temperature (K)":        300.0,
        }

        self.output_columns = [
            "Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)",
            "Velocity[k] (m/s)", "Absolute Pressure (Pa)", "Temperature (K)"
        ]

            # Validate all names exist
        missing = [c for c in self.output_columns if c not in self.freestream_by_name]
        if missing:
            raise ValueError(f"Missing freestream values for columns: {missing}")
        # Ordered vector matching self.output_columns
        fs_vec = np.array([self.freestream_by_name[c] for c in self.output_columns], dtype=float)
        

        return data - fs_vec.reshape(1, -1)
