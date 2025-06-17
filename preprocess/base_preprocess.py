class BasePreprocess:
    def __init__(self, param_files=None, radius_files=None, dimension=3, output_path="processed_data.npz"):
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

        # ``radius_files`` was the original argument name used throughout the
        # code base.  Allow either ``param_files`` or ``radius_files`` to be
        # supplied for backwards compatibility.
        if radius_files is not None and param_files is None:
            param_files = radius_files
        self.files = param_files if param_files is not None else {}
        self.output_path = output_path
        
        self.coords = []
        self.radii = []
        self.outputs = []
        self.npts_max = 0  # max elements determined based on data
        self.dimension = dimension
        self.lhs_applied = False

