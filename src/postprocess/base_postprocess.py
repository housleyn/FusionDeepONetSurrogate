import pandas as pd
import yaml
class BasePostprocess:
    def __init__(self, config_path, path_true=None, path_pred=None):

        if path_true is None:
            self.df_true = None
        else:
            self.df_true = pd.read_csv(path_true)
        df_pred = pd.read_csv(path_pred)
        self.df_pred = df_pred[["Density (kg/m^3)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
                        "Absolute Pressure (Pa)", "Temperature (K)", "X (m)", "Y (m)", "Z (m)"]]
        self.fields = ["Velocity[i] (m/s)", "Velocity[j] (m/s)",
                "Absolute Pressure (Pa)", "Density (kg/m^3)", "Temperature (K)"]
        self.errors = {}
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.project_name = config.get("project_name")
        self.model_type = config.get("model_type")
        self.x_lim = config.get("x_lim", None)
        self.y_lim = config.get("y_lim", None)
        self.param_columns = config.get("param_columns", [])
