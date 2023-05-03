
import numpy as np
import pandas as pd
from scipy import signal
from utils.general import bandpass_filt, shannon_energy

def analyze_vitals(path):
    df = pd.read_parquet(path)[['t', 'x', 'y', 'z']]
    df.set_index('t', inplace=True)
    
    df_down = df.resample('1ms').mean().dropna()
    fs = 1e-09**-1/np.median(np.diff(df_down.index.values.astype(np.int64)))
    
    
    # Calculating HR from MA Z signals
    ma_z_lp = bandpass_filt(df_down['z'].to_frame(), 1/0.05, fs, 'lowpass')
    ma_z_bp = bandpass_filt(ma_z_lp, (20, 51), fs, 'bandpass')

    ma_z_bp_cwt = signal.cwt(ma_z_bp['z'].values.T,signal.morlet,np.arange(1,20))
    ma_z_cwt_pk = ma_z_bp_cwt.T[:,1:15]
    
    
    return df_down

def body_orientations(df):
    gprime = df.values
    (n, _) = gprime.shape
    g0 = np.linalg.norm(gprime, axis=1).reshape(1, -1)
    g = np.hstack([np.zeros((n, 2)), g0.T])
    k = np.cross(g, gprime)
    k /= np.linalg.norm(k, axis=1).reshape(-1, 1)
    
    Rs = R.from_mrp(k)
    return Rs, Rs.as_euler('xyz', degrees=True)