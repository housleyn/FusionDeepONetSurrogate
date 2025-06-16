class BasePreprocess:
    def __init__(self, ellipse_files, dimension, output_path="processed_data.npz"):
        self.files = ellipse_files
        self.output_path = output_path
        
        self.coords = []
        self.radii = []
        self.outputs = []
        self.npts_max = 0  # max elements determined based on data
        self.dimension = dimension
        self.lhs_applied = False
