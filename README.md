Your CSV files must have columns with the following exact titles:

"X (m)"
"Y (m)"
"Z (m)"
"Density (kg/m^3)"
"Velocity[i] (m/s)"
"Velocity[j] (m/s)"
"Velocity[k] (m/s)"
"Absolute Pressure (Pa)"

Your parameter column headers (either geometry or flow paramters) must follow a similar format:

"param1"
"param2"
etc.

your parameters must also be specified in your SurrogateParameters.txt so the surrogate model knows which columns are paramters. They must be passed in as a list:

["param1","param2"]