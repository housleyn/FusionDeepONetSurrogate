import pandas as pd
import numpy as np
from scipy.stats import qmc 
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import os

class MethodsPreprocess:

    def load_data(self):
        for path in self.files:
            print(f"Loading data from {path}")
            df = pd.read_csv(path)
            coords = df[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
            outputs = df[["Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)", "Absolute Pressure (Pa)", "Temperature (K)"]].to_numpy()
            sdf = df[self.distance].to_numpy()
            param_vec = df[self.param_columns].to_numpy()
            if param_vec.shape[0] != coords.shape[0]:
                param_vec = np.repeat(param_vec[:1], coords.shape[0], axis=0) 
            self.coords.append(coords)
            self.params.append(param_vec)
            self.outputs.append(outputs)
            self.sdf.append(sdf)


    def load_and_pad(self):
        self.load_data()
        self.npts_max = max(c.shape[0] for c in self.coords)
        padded_coords = []
        padded_params = []
        padded_outputs = []
        padded_sdf = []

        for c, r, o, s in zip(self.coords, self.params, self.outputs, self.sdf):
            c_pad = self._pad(c)
            r_pad = self._pad(r)
            o_pad = self._pad(o)
            s_pad = self._pad(s)
            padded_coords.append(c_pad)
            padded_params.append(r_pad)
            padded_outputs.append(o_pad)
            padded_sdf.append(s_pad)

        self.sdf = padded_sdf
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
            coords=X_coords,           
            outputs=Y_outputs,         
            params=G_params,           
            sdf=S_sdf,              
            outputs_mean=self.outputs_mean,   
            outputs_std=self.outputs_std,     

        )

    def run_all(self, overwrite=False):
        if overwrite or not os.path.exists(self.output_path):
            self.load_and_pad()
            print("saving preprocessed data to", self.output_path)
            self.save()
        else:
            print(f"Preprocessed file {self.output_path} already exists. Skipping preprocessing.")

    def run_all_low_fi(self, overwrite=False):
        if overwrite or not os.path.exists(self.output_path):
            self.load_and_pad()
            print("saving preprocessed low fidelity data to", self.output_path)
            self.save()
        else:
            print(f"Preprocessed file {self.output_path} already exists. Skipping preprocessing.")

    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  
        return (data - mean) / std, mean, std

    
