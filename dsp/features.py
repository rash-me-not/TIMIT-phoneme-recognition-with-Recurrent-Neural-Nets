'''
@author: mroch
'''

# Standard Python
import sys
import os
import hashlib  # hash functions

# Add ons
import matplotlib.pyplot as plt
import numpy as np

# Our modules
from .audioframes import AudioFrames
from .dftstream import DFTStream
from .utils import spectrogram
from .utils import plot_matrix

from dsp.pca import PCA

class Features:
    """
    Features - class for extracting features from files
    Uses a cache directory if specified.  When feature extraction
    is requested, we see if the features exist and read them.  Otherwise
    features are created and saved (if the cache is enabled).
    
    CAVEATS:
    If feature extraction parameters change, it IS THE USER'S RESPONSIBILITY
    to select a new cache directory or the old parameters will be used.
    No checking is done to see if the extraction parameters match.
    """
    
    ms_per_s = 1000.0   # milliseconds per second
    
    def __init__(self, adv_ms, len_ms, audioroot,
                 specfmt="Mel", pca=None, pca_axes_N=None, 
                 mel_filters_N=15, verbosity=0):
        """
        Spectral features are extracted based on framing parameters.
         
        adv_ms, len_ms - Frame advance and length in ms
        audioroot - Root directory.  All audio files are specified
            relative to this.
        
        Optional arguments:
        
        specfmt = DFTStream format type, e.g. "dB", "Mel", see DFTStream for 
        details
    
        pca, pca_axes_N
        These spectra are projected into a PCA space of the specified number
        of pca_axes_N using the PCA space contained in object pca which is of
        type dsp.pca.PCA.
        
        mel_filters
        Only has an effect when specfmt="Mel".  Specifies the number of Mel
        filters to use.
        
        verbosity - Provide information about processing when > 0
        
        CAVEATS:  Assumes that all audio data have the same sample rate.
        """
        
        # Store parameters for when user wants feature data
        self.adv_ms = adv_ms
        self.len_ms = len_ms
        self.audioroot = audioroot
        self.specfmt = specfmt
        self.pca = pca
        self.pca_axes_N = pca_axes_N
        self.mel_filters_N = mel_filters_N
        self.verbosity = verbosity
        self.cachedir = None
        
        
        if not os.path.isdir(audioroot):
            raise ValueError("audioroot='%s' must be a directory"%(audioroot))
        
     
    def get_advance_s(self):
        "get_advance_s() - Return frame advance in seconds"
        return self.adv_ms / self.ms_per_s
    
    def get_len_s(self):
        "get_len_s() - Return frame length in seconds"
        return self.len_ms / self.ms_per_s
    
    def set_cacheroot(self, cacheroot):
        """"set_cacheroot(cacheroot)
        Use the specified directory (it must exist and be writable) as the root
        of a feature cache system. A subdirectory is created corresponding to
        the current feature set.  When features are requested, the system first
        looks in the cache directory.  If found, the features are returned.
        Otherwise, they are computed, cached in the cache directory, and
        returned.
        
        CAVEATS:
        
        Note that there is no checking for stale features.  If the generation
        code changes, the cache must be manually deleted.
        
        When principal components analysis is used, the cache entries are
        dependent on a hash of the principal component directions and strengths
        (eigen vectors and values).  In the unlikely case of a hash collision,
        incorrect features could be loaded.
        """ 
        
        if not os.path.isdir(cacheroot):
            raise RuntimeError("Cache system:  Directory %s does not exist"%(
                cacheroot))
        if not os.access(cacheroot, os.W_OK):
            raise RuntimeError("Cache system:  Directory %s not writable"%(
                cacheroot))            
        
        # Construct strings detailing filtering/PCA options
        
        specdetails = []
        if self.specfmt == "Mel":   # Mel filtering
            specdetails.append(str(self.mel_filters_N))
            
        if self.pca is not None:  
            # PCA analysis - hash eigen vectors and values
            # to create a key for this feature extraction
            eigvec = self.pca.get_pca_directions()
            eigval = self.pca.get_eigen_values()
            tohash = np.hstack(eigvec, eigval)  # combine into single matrix
            
            md5 = hashlib.md5()  # Message digest hash
            # Hash a buffer containing the byte representation of
            # the eigen vector and value data
            md5.update(tohash.tobytes())  
            hashkey = md5.hexdigest()  # encode key as hexadecimal string
            specdetails.append("PCA" + hashkey)
                        
        # Construct the cache name
        cache = []  # list of cache components
        # Windows systems have a drive letter and :, get rid of it
        # May create problems if there are different directories of the same
        # path on different drives, but this is unlikely
        _, path = os.path.splitdrive(self.audioroot)         
        # Split out the path
        components = []
        head = path
        while len(head) > 0 and head != os.sep:
            head, tail = os.path.split(head)
            components.insert(0, tail)  # Add to front of list                                              
        cache.append("~".join(components))
                     
        cache.append("adv%d"%(self.adv_ms))
        cache.append("len%d"%(self.len_ms))
        cache.append(self.specfmt)
        cache.extend(specdetails)  # extend the list for specialized types
        
        dirname = "_".join(cache)  # glue together to form a directory name
                
        self.cachedir = os.path.join(cacheroot, dirname)
        
    def get_cachedir(self):
        "get_cachedir() - Report the current cache directory (possibly None)"
        return self.cachedir
            
    def get_features(self, relfile):
        """get_features(relfile)
        Given an audio file relative to the audioroot, extract features 
        based on the current settings of the Feature object.
        
        Returns a 2d tensor where each row contains features for a frame
        of the audio data.
        
        CAVEAT:  If caching has been enabled and a cache file exists,
        the cached version will be used EVEN IF THE FEATURE EXTRACTION
        PARAMETERS HAVE CHANGED.
        """
        
        base, _ext = os.path.splitext(relfile)
        
        # Assume we have to generate features until we learn otherwise
        generate = True 
        
        if self.cachedir is not None:
            # Try to read from the cache
            cachefile = os.path.join(self.cachedir, base + ".npz")
            try:
                features = np.load(cachefile)
                generate = False    # Yippee ki-yay!  Happy lazy animal...
            except IOError:
                pass
        
        if generate:
            fullfile = os.path.join(self.audioroot, relfile)
            [features, _t, _f] = spectrogram(
                    [fullfile], self.adv_ms, self.len_ms, 
                    self.specfmt, mel_filters_N = self.mel_filters_N)
            
            if self.pca is not None:
                features = self.pca.transform(features, self.pca_axes_N)
                 
            if self.cachedir:
                # Generated the features, save them
                dirpath, _file = os.path.split(cachefile)
                if not os.path.isdir(dirpath):
                    # Directory path does not exist, create any needed elements
                    os.makedirs(dirpath)                                
                np.save(cachefile, features)
            
        return features
            
            
        
        
     