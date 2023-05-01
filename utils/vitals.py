from general import file_to_lazy_frame
import numpy as np

def body_orientations(df):
    gprime = df.values
    (n, _) = gprime.shape
    g0 = np.linalg.norm(gprime, axis=1).reshape(1, -1)
    g = np.hstack([np.zeros((n, 2)), g0.T])
    k = np.cross(g, gprime)
    k /= np.linalg.norm(k, axis=1).reshape(-1, 1)
    
    Rs = R.from_mrp(k)
    return Rs, Rs.as_euler('xyz', degrees=True)