'''
@author: mroch
'''

from .pca import PCA
from .multifileaudioframes import MultiFileAudioFrames
from .dftstream import DFTStream
from .rmsstream import RMSStream
from .audioframes import AudioFrames
 

# Standard Python libraries
import os.path
from datetime import datetime
# Add-on libraries
import numpy as np
import matplotlib.pyplot as plt


import hashlib  # hash functions
from librosa.feature.spectral import melspectrogram
from statsmodels.tsa.x13 import Spec
from .endpointer import Endpointer

def s_to_frame(s, adv_ms):
    """s_to_frame(s, adv_ms) 
    Convert s in seconds to a frame index assuming a frame advance of adv_ms
    """
    
    return np.int(np.round(s * 1000.0 / adv_ms))

def plot_matrix(matrix, xaxis=None, yaxis=None, xunits='time (s)', yunits='Hz', zunits='(dB rel.)'):
    """plot_matrix(matrix, xaxis, yaxis, xunits, yunits
    Plot a matrix.  Label columns and rows with xaxis and yaxis respectively
    Intensity map is labeled zunits.
    Put "" in any units field to prevent plot of axis label
    
    Default values are for an uncalibrated spectrogram and are inappropriate
    if the x and y axes are not provided
    """
    
    if xaxis is None:
        xaxis = [c for c in range(matrix.shape[1])]
    if yaxis is None:
        yaxis = [r for r in range(matrix.shape[0])]
        
    # Plot the matrix as a mesh, label axes and add a scale bar for
    # matrix values
    plt.pcolormesh(xaxis, yaxis, matrix)
    plt.xlabel(xunits)
    plt.ylabel(yunits)
    plt.colorbar(label=zunits)
    
def spectrogram(files, adv_ms, len_ms, specfmt="dB", mel_filters_N=12):
    """spectrogram(files, adv_ms, len_ms, specfmt)
    Given a filename/list of files and framing parameters (advance, length in ms), 
    compute a spectrogram that spans the files.
    
    Type of spectrogram (specfmt) returned depends on DFTStream, see class
    for valid arguments and interpretation, defaults to returning
    intensity in dB.
    
    Returns [intensity, taxis_s, faxis_Hz]
    """

    # If not a list, make it so number one...
    if not isinstance(files, list):
        files = [files]
        
    # Set up frame stream and pass to DFT streamer
    framestream = MultiFileAudioFrames(files, adv_ms, len_ms)
    dftstream = DFTStream(framestream, specfmt=specfmt, mels_N = mel_filters_N)
    
    # Grab the spectra
    spectra = []
    for s in dftstream:
        spectra.append(s)
        
    # Convert to matrix
    spectra = np.asarray(spectra)
        
    # Time axis in s
    adv_s = framestream.get_frameadv_ms() / 1000    
    t = [s * adv_s for s in range(spectra.shape[0]) ]
    
    return [spectra, t, dftstream.get_Hz()]


def fixed_len_spectrogram(file, adv_ms, len_ms, offset_s, specfmt="dB", 
                          mel_filters_N = 12):
    """fixed_len_spectrogram(file, adv_ms, len_ms, offset_s, specfmt, 
        mel_filters_N)
        
    Generate a spectrogram from the given file.
    Truncate the spectrogram to the specified number of seconds
    
    adv_ms, len_ms - Advance and length of frames in ms
    
    offset_s - The spectrogram will be truncated to a fixed duration,
        centered on the median time of the speech distribution.  The
        amount of time to either side is determned by a duration in seconds,
        offset_s.  
        
        The speech is endpointed using an RMS energy endpointer
        and centered median time of frames marked as speech.  If the fixed
        duration is longer than the available speech, random noise frames
        are drawn from sections marked as noise to complete the spectrogram
        
    specfmt - Spectrogram format. See dsp.dftstream.DFTStream for valid formats
    
    mel_filters_N - Number of Mel filters to use when specft == "Mel"
    """
    
    
    endpointer = Endpointer(file, adv_ms, len_ms)
    # Get logical indexing array with speech frames marked as True
    speech_indicator = endpointer.speech_frames()
    framesN = np.product(speech_indicator.shape)
    
    # Create time axis and determine where the center of the speech is
    # We'll use median to avoid undue influence from outliers.
    adv_s = adv_ms / 1000.0
    taxis = np.asarray([t * adv_s for t in range(framesN)])
    center_s = np.median(taxis[speech_indicator])
    
    # We now need to retain speclen_s seconds of speech, 
    # centered on the central point of the speech.  Depending on the length
    # this might go past either end of the audio data.
    # Note that we use the duration rather than the right side to ensure
    # that rounding does not produce an inconsistent number of frames
    left_idx = s_to_frame(center_s - offset_s, adv_ms)
    frames = s_to_frame(2*offset_s, adv_ms)
    right_idx = left_idx + frames
    # Fill needed?
    fill_left = np.abs(left_idx) if left_idx < 0 else 0
    unused_right = len(speech_indicator) - right_idx
    if unused_right < 0:
        fill_right = right_idx - framesN
    else:
        fill_right = 0
    # Fix left and right indices if needed, as they might have
    # gone past the end of the data
    left_idx = np.max([0, left_idx])
    right_idx = np.min([framesN, right_idx])    
    
    
    # Compute spectrogram
    [spec, spec_t, spec_Hz] = \
        spectrogram([file], adv_ms, len_ms, specfmt=specfmt)

        
    # Truncate/fill
    # We will fill truncated regions with zeros.  This is convenient,
    # but not a great strategy if we were writing production-level code
    # as it is possible that the net will simply learn that a bunch of
    # zeros at the front/end imply specific classes.  This isn't totally
    # inappropriate as the net would be learning that certain classes
    # are of shorter duration.  We will learn more sophisticated ways of 
    # dealing with sequence data later in the semester. 
    if fill_left > 0:
        left_pad = np.zeros([fill_left, spec.shape[1]])
        feature_matrix = np.vstack((left_pad, spec[left_idx:right_idx,:]))
    else:
        feature_matrix = spec[left_idx:right_idx,:]
    if fill_right > 0:
        right_pad = np.zeros([fill_right, spec.shape[1]])
        feature_matrix = np.vstack((feature_matrix, right_pad))
    
    # Compute truncated time axis
    offset_s = center_s - offset_s  # Might start at < 0, okay
    time_s = [t*adv_s + offset_s for t in range(feature_matrix.shape[0])]
    return [feature_matrix, time_s, spec_Hz]

    
    
def pca_analysis_of_spectra(files, adv_ms, len_ms, offset_s): 
    """"pca_analysis_of_spectra(files, advs_ms, len_ms, offset_s)
    Conduct PCA analysis on spectra of the given files
    using the given framing parameters.  Only retain
    central -/+ offset_s of spectra
    """

    md5 = hashlib.md5()
    string = "".join(files)
    md5.update(string.encode('utf-8'))
    hashkey = md5.hexdigest()
    
    filename = "VarCovar-" + hashkey + ".pcl"
    try:
        pca = PCA.load(filename)

    except FileNotFoundError:
        example_list = []
        for f in files:
            [example, _t, _f] = fixed_len_spectrogram(f, adv_ms, len_ms, offset_s, "dB")
            example_list.append(example)
            
        # row oriented examples
        spectra = np.vstack(example_list)
    
        # principal components analysis
        pca = PCA(spectra)

        # Save it for next time
        pca.save(filename)
        
    return pca


       
def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    """
    
    files = []
    
    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))
                         
    return files
    
def get_class(files):
    """get_class(files)
    Given a list of files, extract numeric class labels from the filenames
    """
    
    # TIDIGITS single digit file specific
    
    classmap = {'z': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'o': 10}

    # Class name is given by first character of filename    
    classes = []
    for f in files:        
        dir, fname = os.path.split(f) # Access filename without path
        classes.append(classmap[fname[0]])
        
    return classes
    
class Timer:
    """Class for timing execution
    Usage:
        t = Timer()
        ... do stuff ...
        print(t.elapsed())  # Time elapsed since timer started        
    """
    def __init__(self):
        "timer() - start timing elapsed wall clock time"
        self.start = datetime.now()
        
    def reset(self):
        "reset() - reset clock"
        self.start = datetime.now()
        
    def elapsed(self):
        "elapsed() - return time elapsed since start or last reset"
        return datetime.now() - self.start
    
