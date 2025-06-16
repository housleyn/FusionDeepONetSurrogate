class BasePreprocess:
    def __init__(self, radius_files, dimension, output_path="processed_data.npz"):
        """Initialize the preprocessing object.

        Parameters
        ----------
        radius_files : dict
            Mapping from radius value to CSV file path.
        dimension : int
            Dimensionality of the data (2 or 3).
        output_path : str, optional
            Where the processed ``npz`` file will be written.
        """

        self.files = radius_files
        self.output_path = output_path
        
        self.coords = []
        self.radii = []
        self.outputs = []
        self.npts_max = 0  # max elements determined based on data
        self.dimension = dimension
        self.lhs_applied = False
