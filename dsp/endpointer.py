'''
@author: mroch

endpointer - Module for unsupervised learning of speech and noise

Endpointing is implemented via RMS energy and 2 mixture Guassian mixture
models
'''

import numpy as np
from sklearn.mixture import GaussianMixture


from .audioframes import AudioFrames
from .rmsstream import RMSStream


class Endpointer:
    
    def __init__(self, filename, adv_ms, len_ms):
        """
        Endpointer(filename, adv_ms, len_ms)
        Speech/noise detection of audio data contained in filename
        
        filename - file to be analyzed
        adv_ms, len_ms - frame parameters in milliseconds        
        """
        
        # Create an root-mean-square energy frame stream
        framer = AudioFrames(filename, adv_ms, len_ms)
        rms_stream = RMSStream(framer)
        
        # save framing parameters for posterity
        self.adv_ms = adv_ms
        self.len_ms = len_ms

        # Column vector of intensities        
        self.intensity_dB = np.asarray([dB for dB in rms_stream])
        self.intensity_dB = self.intensity_dB.reshape([-1, 1])
        
        # Train Gaussian mixture model
        mixtures = 2    # assume two classes
        self.gmm = GaussianMixture(mixtures)
        self.gmm.fit(self.intensity_dB)
        
        # determine speech and noise categories
        # predictions are category 0 or 1.  Find out which one is noise and
        # which one is speech
        self.noise_category = np.argmin(self.gmm.means_)  # smaller mean is quieter      
        # Other one is speech
        self.speech_category = (self.noise_category + 1) % 2        
        
        # See what categories our unsupervised learner assigns
        # to the clustered data
        self.predictions = self.gmm.predict(self.intensity_dB)
        
    def get_speech_noise_labels(self):
        "Return tuple indicating noise and speech categories: (noise, speech)"
        
        return (self.noise_category, self.speech_category)
    
    def speech_frames(self):
        """"speech_frames()
        Return indicator vector that can be used for logical indexing
        True is speech,"""
        return self.predictions == self.speech_category
    
    def noise_frames(self):
        """"noise_frames()
        Return indicator vector that can be used for logical indexing
        True is noise,"""
        return self.predictions == self.noise_category
    

