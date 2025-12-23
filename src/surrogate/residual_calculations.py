import torch
from src.models.low_fi_fusion_model import Low_Fidelity_FusionDeepONet
import numpy as np

def make_residual_dataset(self, hf_npz_out, low_fi_stats_path, high_fi_stats_path):
        hf_train_loader = self.train_loader
        hf_test_loader  = self.test_loader

        lf_stats = self._load_stats(low_fi_stats_path)
        hf_stats = self._load_stats(high_fi_stats_path)
        mu_lf, std_lf = lf_stats["outputs_mean"].to(self.device), lf_stats["outputs_std"].to(self.device)
        mu_hf, std_hf = hf_stats["outputs_mean"].to(self.device), hf_stats["outputs_std"].to(self.device)

        lf_model = Low_Fidelity_FusionDeepONet(
            coord_dim=self.coord_dim + self.distance_dim,
            param_dim=self.param_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            out_dim=self.output_dim,
            npz_path=self.low_fi_output_path
        ).to(self.device)
        lf_model.load_state_dict(torch.load(self.low_fi_model_path, map_location=self.device))
        lf_model.eval()

        all_coords, all_params, all_sdf = [], [], []
        all_uHF, all_uLF, all_residual = [], [], []

        with torch.no_grad():
            for loader in [hf_train_loader, hf_test_loader]:
                for batch in loader:
                    if isinstance(batch, dict):
                        coords = batch["coords"].to(self.device)
                        params = batch["params"].to(self.device)
                        outputs = batch["outputs"].to(self.device)
                        sdf     = batch["sdf"].to(self.device)
                    else:
                        coords, params, outputs, sdf = [t.to(self.device) for t in batch]
                    targets = outputs  

                    u_hf_denorm = targets * std_hf + mu_hf
                    u_lf = lf_model(coords, params, sdf)
                    assert u_lf.shape == targets.shape, \
                        f"LF/HF shape mismatch: lf {u_lf.shape}, hf {targets.shape}"
                    u_lf_denorm = u_lf * std_lf + mu_lf
                    
                    r_denorm = u_hf_denorm - u_lf_denorm

                    all_coords.append(coords.cpu().numpy())
                    all_params.append(params.cpu().numpy())
                    all_sdf.append(sdf.cpu().numpy())
                    all_uHF.append(u_hf_denorm.cpu().numpy())
                    all_uLF.append(u_lf_denorm.cpu().numpy())
                    all_residual.append(r_denorm.cpu().numpy())

        
        def cat(lst): return np.concatenate(lst, axis=0) if len(lst) > 1 else lst[0]

        residuals = cat(all_residual)       
        uLF = cat(all_uLF)                  
        uHF = cat(all_uHF)
        coords = cat(all_coords)
        params = cat(all_params)
        sdf = cat(all_sdf)

        
        residuals_flat = residuals.reshape(-1, residuals.shape[-1])
        mu_r = torch.tensor(np.mean(residuals_flat, axis=0), dtype=torch.float32)
        std_r = torch.tensor(np.std(residuals_flat, axis=0) + 1e-8, dtype=torch.float32)

        
        r_norm = (residuals - mu_r.numpy()) / std_r.numpy()
        uLF_norm = (uLF - mu_r.numpy()) / std_r.numpy()

        np.savez_compressed(
            hf_npz_out,
            coords=coords,
            params=params,
            sdf=sdf,
            outputs=r_norm,
            aux_lf_pointwise=uLF_norm,
            targets_highfi=uHF,
            outputs_mean=mu_r.numpy(),   
            outputs_std=std_r.numpy()
        )
        print(f"Residual dataset written to: {hf_npz_out}")