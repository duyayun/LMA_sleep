
import numpy as np
import mne
import pandas as pd
import polars as pl
from scipy import signal

def file_to_lazy_frame(filename):
    extension = filename.split('.')[-1]
    if extension == 'csv':
        ret = pl.read_csv(filename, columns=['t', 'x', 'y', 'z'], use_pyarrow=True).select([
            pl.col('ts').cast(pl.Datetime), 
            pl.col('x').cast(pl.Int16),
            pl.col('y').cast(pl.Int16),
            pl.col('z').cast(pl.Int16)
            ]).interpolate().filter(pl.col('ts').is_not_null()).filter(pl.col('z').is_not_null())
        
    elif extension == 'parquet':
        ret = pl.read_parquet(filename, columns=['t', 'x', 'y', 'z'], use_pyarrow=True).select([
            pl.col('t').cast(pl.Datetime), 
            pl.col('x').cast(pl.Int16),
            pl.col('y').cast(pl.Int16),
            pl.col('z').cast(pl.Int16)
            ]).interpolate().filter(pl.col('t').is_not_null()).filter(pl.col('z').is_not_null())
    elif extension == 'gz':
        print('gz format is not yet supported')

    else:
        print(f'bad file format: {extension}')    
    
    
    return ret



def shannon_energy(x):
    """Implementation of shannon energy

    Args:
        x (_type_): input signal
    """
    x_env = -x**2 * np.log(x**2)
    return(x_env)


def bandpass_filt(sig,cutoff,fs,mode):
    """ bandpass filtering

    Args:
        sig (_type_): input signal. 
        cutoff (_type_): cutoff requency
        fs (_type_): signal frequency
        mode (_type_): choice between 'lowpass' 'highpass' and 'bandpass'

    Returns:
        _type_: filtered signal
    """
    nyq = 0.5*fs
    wn = cutoff/nyq
    sos = signal.butter(4, wn, btype=mode,output='sos')
    return pd.DataFrame(data=signal.sosfiltfilt(sos,sig,axis=0),columns=sig.columns,index=sig.index)


def read_edf(path):
    """reads edf file into dataframe, and upsample to 1kHz
    

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    raw = mne.io.read_raw_edf(path)
    start_time = raw.info['meas_date'].replace(tzinfo=None)
    df = raw.to_data_frame()
    df['time'] = pd.to_timedelta(df['time'], unit='s') + start_time
    df.set_index('time', inplace=True)
    df = df[['PR', 'RR']]
    df_resampled = df.resample('1ms').asfreq()
    df_upsampled = df_resampled.interpolate(method='linear')
    
    df_upsampled['PR'] *= 1e-6

    return df_upsampled




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