class BasePreprocess:
    def __init__(self, files, dimension, output_path, param_columns, lhs_sample=100):


        self.files = files
        self.output_path = output_path
        self.param_columns = param_columns
        self.coords = []
        self.params = []
        self.outputs = []
        self.sdf = []
        self.npts_max = 0  
        self.dimension = dimension
        self.lhs_applied = False
        self.lhs_sample = lhs_sample
        self.distance = "distanceToEllipse"
