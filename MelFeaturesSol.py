import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

def hann_window(N):
    """
    Create the Hann window 0.5*(1-cos(2pi*n/N))
    """
    return 0.5*(1 - np.cos(2*np.pi*np.arange(N)/N))

def specgram(x, win_length, hop_length, win_fn = hann_window):
    """
    Compute the non-redundant amplitudes of the STFT
    Parameters
    ----------
    x: ndarray(N)
        Full audio clip of N samples
    win_length: int
        Window length to use in STFT
    hop_length: int
        Hop length to use in STFT
    win_fn: int -> ndarray(N)
        Window function
    
    Returns
    -------
    ndarray(w, floor(w/2)+1, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-win_length)/hop_length))+1
    K = int(np.floor(win_length/2))+1
    # Make a 2D array
    # The rows correspond to frequency bins
    # The columns correspond to windows moved forward in time
    S = np.zeros((K, nwin))
    # Loop through all of the windows, and put the fourier
    # transform amplitudes of each window in its own column
    for j in range(nwin):
        # Pull out the audio in the jth window
        xj = x[hop_length*j:hop_length*j+win_length]
        # Zeropad if necessary
        if len(xj) < win_length:
            xj = np.concatenate((xj, np.zeros(win_length-len(xj))))
        # Apply window function
        xj = win_fn(win_length)*xj
        # Put the fourier transform into S
        sj = np.abs(np.fft.fft(xj))
        S[:, j] = sj[0:K]
    return S

def get_mel_spectrogram(K, win_length, sr, min_freq, max_freq, n_bins):
    """
    Compute a mel-spaced spectrogram by multiplying an linearly-spaced
    STFT spectrogram on the left by a mel matrix
    
    Parameters
    ----------
    win_length: int
        Window length
    K: int
        Number of frequency bins
    sr: int
        The sample rate used to generate sdb
    min_freq: int
        The center of the minimum mel bin, in hz
    max_freq: int
        The center of the maximum mel bin, in hz
    n_bins: int
        The number of mel bins to use
    
    Returns
    -------
    ndarray(n_bins, n_win)
        The mel-spaced spectrogram
    """
    bins = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bins+2)*win_length/sr
    bins = np.array(np.round(bins), dtype=int)
    Mel = np.zeros((n_bins, K))
    for i in range(n_bins):
        i1 = bins[i]
        i2 = bins[i+1]
        if i1 == i2:
            i2 += 1
        i3 = bins[i+2]
        if i3 <= i2:
            i3 = i2+1
        tri = np.zeros(K)
        tri[i1:i2] = np.linspace(0, 1, i2-i1)
        tri[i2:i3] = np.linspace(1, 0, i3-i2)
        #tri = tri/np.sum(tri)
        Mel[i, :] = tri
    return Mel

def get_mfcc(x, sr, win_length=2048, hop_length=512, min_freq=80, max_freq=8000, n_bins=100, n_coeffs=20, amin=1e-5):
    """
    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate
    win_length: int
        Window length to use in STFT
    hop_length: int
        Hop length to use in STFT
    min_freq: float
        Minimum frequency, in hz, to use in mel-spaced bins
    max_freq: float
        Maximum frequency, in hz, to use in mel-spaced bins
    n_bins: int
        Number of bins to take between min_freq and max_freq
    n_coeffs: int
        Number of DCT coefficients to use in the summary
    amin: float
        Minimum threshold for integrated energy
    """
    S = specgram(x, win_length, hop_length)
    S = np.abs(S)**2
    Mel = get_mel_spectrogram(S.shape[0], sr, 80, 8000, 40)
    mfcc = Mel.dot(S)
    mfcc[mfcc < amin] = amin
    mfcc = np.log10(mfcc)
    mfcc = dct(mfcc, axis=0)
    return mfcc[0:n_coeffs, :]
