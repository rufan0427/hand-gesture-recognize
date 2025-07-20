import h5py

with h5py.File('./archive1/hand_landmarks_dataset_train2.h5', 'r') as f:
    print("Keys in the HDF5 file:", list(f.keys()))
    print("Dataset structure:")
    for key in f.keys():
        print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

    for i in range(len(f['keypoints'][1])):
        print(f"Keypoint {i}: {f['keypoints'][1][i]}")