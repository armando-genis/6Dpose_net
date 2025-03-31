import h5py
import numpy as np
keys = []
ids = []
with h5py.File("efficient_pose.h5",'r') as f: # open file
    f.visit(keys.append) # append all keys to list
    for id, key in enumerate(keys):
        
        if ':' in key: # contains data if ':' in key
            if ('model_weights' in key):
                print(f[key].name)
                print (np.array(f[key]))
                ids.append(id)
            

group = keys[ids[0]]
f = h5py.File("efficient_pose.h5",'r')
print (np.array(f[group]))
print (f)
print (f['model_weights'])
