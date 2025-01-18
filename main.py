# imports
import h5py
import numpy as np


# exploring hdf5 file
def checkfile(filename):
    try:
        with h5py.File(filename, 'r') as f:
            print(list(f.keys()))
    except Exception as e:
        print(e)

# # reading hdf5 file
# for i in range(50):
#     checkfile(f'transfer cube human data/episode_{i}.hdf5')


# Function to explore the contents of a single HDF5 file
def explore_hdf5_item(item, indent=0):
    if isinstance(item, h5py.Dataset):
        print(f"{' ' * indent}Dataset: {item.name}")
        print(f"{' ' * indent}Shape:", item.shape)
        print(f"{' ' * indent}Data type:", item.dtype)

        # Print a sample of the data
        if item.shape[0] > 0:
            print(f"{' ' * indent}Sample data:", item[:min(5, item.shape[0])])
        else:
            print(f"{' ' * indent}Dataset is empty.")
    elif isinstance(item, h5py.Group):
        print(f"{' ' * indent}Group: {item.name}")
        for key, sub_item in item.items():
            explore_hdf5_item(sub_item, indent + 2)

# Function to explore the contents of a single HDF5 file
def explore_single_hdf5_file(filename):
    try:
        with h5py.File(filename, 'r') as f:
            print(f"Exploring file: {filename}")
            # Print the keys (dataset names) in the file
            print("Keys in the file:", list(f.keys()))

            # Iterate over each key (dataset or group) and print information
            for key, item in f.items():
                explore_hdf5_item(item)

    except Exception as e:
        print(f"Error exploring file {filename}: {e}")

# Example usage
# filename = 'transfer cube human data/episode_0.hdf5'  # Replace with your file path
# explore_single_hdf5_file(filename)


# now exploring data provided for vqbet

# Function to explore the contents of a .npy file
def explore_npy_file(filename):
    try:
        # Load the .npy file
        data = np.load(filename)

        print(f"Exploring file: {filename}")
        print("Shape:", data.shape)
        print("Data type:", data.dtype)

        # Print a sample of the data
        if data.size > 0:
            print("Sample data:", data[:min(5, data.shape[0])])
        else:
            print("Dataset is empty.")

    except Exception as e:
        print(f"Error exploring file {filename}: {e}")

# Example usage
filenames = [
    '/Users/amnesiac/Downloads/vqbet_datasets_for_release/ur3/data_msk.npy',
    '/Users/amnesiac/Downloads/vqbet_datasets_for_release/ur3/data_act.npy',
    '/Users/amnesiac/Downloads/vqbet_datasets_for_release/ur3/data_obs.npy'
]

for filename in filenames:
    explore_npy_file(filename)
    print("\n" + "-"*50 + "\n")  # Separator for better readability