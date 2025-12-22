class BasePreprocess:
    def __init__(self, files, output_path, param_columns, distance_columns):

        self.files = files
        self.output_path = output_path
        self.param_columns = param_columns
        self.coords = []
        self.params = []
        self.outputs = []
        self.sdf = []
        self.npts_max = 0 
        self.distance = distance_columns
