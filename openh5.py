import h5py

with h5py.File('./archive1/process_testing.h5', 'r') as f:
    print("Keys in the HDF5 file:", list(f.keys()))
    print("Dataset structure:")
    for key in f.keys():
        print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")