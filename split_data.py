import splitfolders

input_folder = "dataset/raw/PlantVillage"  # <--- point here
output_folder = "dataset"

# Split 70% train, 20% val, 10% test
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.2, 0.1))
