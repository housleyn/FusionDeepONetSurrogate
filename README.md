#CSV file format
Your CSV files used as training data must have columns with the following exact titles:

"X (m)"
"Y (m)"
"Z (m)"
"Density (kg/m^3)"
"Velocity[i] (m/s)"
"Velocity[j] (m/s)"
"Velocity[k] (m/s)"
"Absolute Pressure (Pa)"

Even if your simulation is only 2D, include a "Z (m)" column with all zeros. Your parameter column headers (either geometry or flow paramters) must follow a similar format:

"param1"
"param2"
etc.

This repository is built to handle all of your parameters, coordinates, density, velocities, and absolute pressure in the same file.

#Work Flow
This code is meant to be simple. All actions can be carried out through the file main.py. For typical use, instantiating a Surrogate object is all that you will need. The work flow for this object is simple:

step 1-instantiate:
    example = Surrogate(files=example.txt)
step 2-train:
    example._train()
step 3-infer:
    example_inference(file=example_file_coords_and_params.csv)
    or
    example_infer_and_validate(file=example_file_with_validating_data)

Step 1_instantiate: In this step you simply create an instance and call it what you would like. The Surrogate object takes in a txt file. This file contains all the settings for the surrogate, an example and explanation will be shown below.

Step 2_train: This command will train the surrogate based on the settings provided. You should note that typically, this step need not be repeated. If you have provided enough data, the surrogate only needs to be trained once, and can then make as many inferences as you'd like.

Step 3_infer: This command will make an inference on the .csv file that you pass in. The format of this csv file is similar to the format of your training data. There are two possible commands to run: infer, and infer with validation. Just infering will simply take your input file and run it through the model and output data and visuals. If you wish to infer and validate, the file you pass in will also contain 'true' data for all of the flow fields, this command will pass your file through the model and compare the surrogate's outputs with the 'true' data. This comparison involves eror plots and tables showing the accuracy of the model.


#input txt file for surrogate object

