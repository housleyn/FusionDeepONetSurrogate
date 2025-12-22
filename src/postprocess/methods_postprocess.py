import numpy as np
import os
from .plotting_postprocess import (create_table, plot_fields, compute_surface_percent_differences, create_surface_metrics_table)



class MethodsPostprocess:
    def run(self, dimension):
        self._calculate_error()
        self._define_output_folders()
        fields = self.fields
        if dimension == 3:
            fields.insert(2, "Velocity[k] (m/s)")
        for field in fields:
            self._calculate_relative_l2_error(field)
        create_table(self)
        plot_fields(self)
        surface_metrics = compute_surface_percent_differences(self)
        create_surface_metrics_table(
            metrics=surface_metrics,
            save_path=os.path.join(self.tables_dir, "surface_metric_errors.png")
        )

    def get_errors(self):
        self._calculate_error()
        fields = self.fields 
        for field in fields:
            self._calculate_relative_l2_error(field)
        return self.errors.items()
    
    def _calculate_error(self):
        error = np.abs(self.df_true - self.df_pred)
        error["X (m)"] = self.df_true["X (m)"]
        error["Y (m)"] = self.df_true["Y (m)"]
        self.error = error

    def _calculate_relative_l2_error(self, field):
        u_true = self.df_true[field].values
        u_pred = self.df_pred[field].values
        rel_l2 = 100 * np.linalg.norm(u_true-u_pred) / np.linalg.norm(u_true)
        self.errors[field] = rel_l2

    def _define_output_folders(self):
        self.figures_dir = os.path.join("Outputs", self.project_name)
        self.tables_dir = os.path.join("Outputs", self.project_name)
        os.makedirs(self.figures_dir, exist_ok=True)

    





    

    

