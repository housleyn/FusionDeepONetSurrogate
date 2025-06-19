import pandas as pd
class BasePostprocess:
    def __init__(self, path_true=None, path_pred=None, param_columns=None):

        if path_true is None:
            self.df_true = None
        else:
            self.df_true = pd.read_csv(path_true).drop(columns=param_columns)
        df_pred = pd.read_csv(path_pred)
        self.df_pred = df_pred[["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
                        "Absolute Pressure (Pa)", "Density (kg/m^3)", "X (m)", "Y (m)", "Z (m)"]]
        self.errors = {}