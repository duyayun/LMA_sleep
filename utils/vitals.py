
import numpy as np
import pandas as pd
from scipy import signal
from utils.general import bandpass_filt, shannon_energy
from scipy import ndimage
def analyze_vitals(path, shift=0):
    extension = path.split('.')[-1]
    if extension == 'csv':
        print('reading csv file')
        df = pd.read_csv(path)[['ts', 'x', 'y', 'z']].rename({'ts':'t'})
    else:
        print('reading parquet file')
        df = pd.read_parquet(path, engine='pyarrow')[['t', 'x', 'y', 'z']]

    df.set_index('t', inplace=True)

    df_down = df.resample('1ms').mean().dropna()
    fs = 1e-09**-1/np.median(np.diff(df_down.index.values.astype(np.int64)))


    # Calculating HR from MA Z signals
    ma_z_lp = bandpass_filt(df_down['z'].to_frame(), 1/0.05, fs, 'lowpass')
    ma_z_bp = bandpass_filt(ma_z_lp, (20, 51), fs, 'bandpass')

    ma_z_bp_cwt = signal.cwt(ma_z_bp['z'].values.T,signal.morlet,np.arange(1,20))
    ma_z_cwt_pk = ma_z_bp_cwt.T[:,1:15]
    cwt_pks = ma_z_cwt_pk.max(axis=1)


    # Shannon energy followed by gaussian_filter on absolute values of shannon energy
    cwt_se = shannon_energy(cwt_pks)
    cwt_se_gs = ndimage.gaussian_filter(np.abs(cwt_se),sigma=25)
    cwt_se_gs = pd.DataFrame({'z':cwt_se_gs}, index=df_down.index)
    single_peak = bandpass_filt(cwt_se_gs, 1/0.7, fs, 'lowpass')
    peaks, _ = signal.find_peaks(single_peak.values.T[0],height=2e-7,distance=350)
    peak_intervals = np.diff(peaks) / fs

    # Dynamic HR calculation

    heart_rates = 60 / peak_intervals
    hr_df = pd.DataFrame({'ma_hr':heart_rates}, index=df_down.index[peaks[:-1]])

    hr_df = bandpass_filt(hr_df, 50, 1000, 'lowpass')
    # hr_df['ma_rr_interval'] = hr_df['ma_hr']/60

    ds_for_orient = df_down.resample('100ms').mean().fillna(method='ffill').fillna(method='bfill')
    phi, _ = resp_orient(ds_for_orient.values.T, fs)
    phi = pd.DataFrame({'resp_orientation':phi}, index=ds_for_orient.index)

    phi = phi.resample('1ms').interpolate(method='linear')

    rr_single_peak = bandpass_filt(phi, np.array([0.05, 0.4]), 1000, 'bandpass')

    rr_peaks, _ = signal.find_peaks(rr_single_peak.values.T[0],height=-0.05, distance=1000)

    rr_peak_intervals = np.diff(rr_peaks) / fs


    # Dynamic HR calculation

    respiration_rate = 60 / rr_peak_intervals
    respiration_rate = pd.DataFrame({'ma_rr':respiration_rate}, index=rr_single_peak.index[rr_peaks[:-1]])
    ret = pd.concat([df_down, hr_df, phi, respiration_rate], axis=1).fillna(method='ffill').fillna(method='bfill')
    ret.index = ret.index - pd.Timedelta(hours=5)

    return ret.shift(shift)



def body_orientations(df):
    gprime = df.values
    (n, _) = gprime.shape
    g0 = np.linalg.norm(gprime, axis=1).reshape(1, -1)
    g = np.hstack([np.zeros((n, 2)), g0.T])
    k = np.cross(g, gprime)
    k /= np.linalg.norm(k, axis=1).reshape(-1, 1)
    
    Rs = R.from_mrp(k)
    return Rs, Rs.as_euler('xyz', degrees=True)


def resp_orient(a,fs):
    """ function that generates orientation phi. Implemented by Andreas Tzavelis following precedures
    described in https://ieeexplore.ieee.org/document/5504743.
 
    Args:
        a (_type_): numpy array containing MA data
        fs (_type_): frequency

    Returns:
        _type_: _description_
    """
    n = a.shape
    
    # normalize each acceleration vector
    a = a/np.sqrt(a[0,:]**2+a[1,:]**2+a[2,:]**2)

    # calulate adjacent rotation angles and vectors
    theta = np.array([np.arccos(np.dot(a[:,i+1],a[:,i])) for i in range(n[1]-1)])
    r = np.array([np.cross(a[:,i+1],a[:,i]) for i in range(n[1]-1)])
    
    # Calculate covariance matrix and PCA
    C = np.cov(r.T)
    eigval, eigvec = np.linalg.eig(C)
    r_ref = eigvec[:,np.argmax(eigval)] # prevailing rotational axis
    
    # force rotation axis into same hemisphere (rotations back and forth would otherwise flip axis)
    r_t = np.array([np.multiply(r[:,i],np.sign(np.dot(r,r_ref))) for i in range(3)])
    
    # Isolate predominant rotational axis at a point by weighting by instantaneous angle and nearby points
    W_norm = int(30*fs)
    h_win = np.hamming(W_norm)
    r_t_norm = np.array([np.convolve(np.multiply(theta,r_t[i,:]),h_win,mode='same') for i in range(3)])
    r_t_norm = r_t_norm/np.sqrt(r_t_norm[0,:]**2+r_t_norm[1,:]**2+r_t_norm[2,:]**2)
    
    # Average acceleration axis (presumed direction of gravity)
    a_ctrl_norm = np.array([np.convolve(a[i,:],np.ones(W_norm)/W_norm,mode='same') for i in range(3)])
    a_ctrl_norm = a_ctrl_norm/np.sqrt(a_ctrl_norm[0,:]**2+a_ctrl_norm[1,:]**2+a_ctrl_norm[2,:]**2)
    
    # calculate rotation angles about gravity axis
    phi = np.array([np.arcsin(np.dot(np.cross(a_ctrl_norm[:,i],r_t_norm[:,i]),a[:,i])) for i in range(n[1]-1)])
    phi = np.pad(phi,(0,1),mode='edge')
        
    return phi, a_ctrl_norm