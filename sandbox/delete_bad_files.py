import os
import re

folder = r"Data/waverider_lf_data"   # change this

pattern = re.compile(r"_z_([0-9.]+)\.csv$")

for file in os.listdir(folder):
    if file.endswith(".csv"):
        match = pattern.search(file)
        if match:
            z_val = float(match.group(1))
            
            if z_val > 4.2:
                filepath = os.path.join(folder, file)
                os.remove(filepath)
                print(f"Deleted: {file}")