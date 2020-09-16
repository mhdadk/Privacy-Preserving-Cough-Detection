import numpy as np
from tensorflow_addons.image import sparse_image_warp

class SpecAugment:
    
    def __init__(self,
                 num_aug,
                 aug_split,
                 num_time_masks,
                 num_frequency_masks,
                 time_warp_param = 30,
                 time_mask_param = 30,
                 frequency_mask_param = 30,
                 num_masks_random = False):
        
        # number of Mel-scale spectrogram augmentations to perform
        
        self.num_aug = num_aug
        
        """
        list of proportions of the the total number of num_aug augmentations.
        
        The first element of this list should indicate the proportion of num_aug
        augmentations that should be time warped. For example, 0.4 indicates
        that 40% of the total number of num_aug augmentations will be time
        warped.
        
        The second element of this list should indicate the proportion of
        num_aug augmentations that should be time masked. For example, 0.6
        indicates that 60% of the total number of num_aug augmentations will
        be time masked.
        
        The third element of this list should indicate the proportion of
        num_aug augmentations that should be frequency masked. For example, 0.5
        indicates that 50% of the total number of num_aug augmentations will
        be frequency masked.
        
        Note that these proportions don't necessarily need to add up to 1. Also,
        since these proportions are achieved through random sampling, then each
        of their values should be at least 0.5 to avoid the case where none of
        these augmentations are applied.
        
        Note that these transformations are not mutually exclusive, which means
        that different combinations of these transformations will be applied to
        some proportion of the total number of num_aug augmentations.
        
        For example, if:
            
        aug_split = [0.4,0.6,0.7]
        
        Then this means that of the total number of num_aug augmentations,
        40% will be time warped, 60% will be time masked, and 70% will be
        frequency masked.
        """
        
        self.aug_split = aug_split
        
        # time warping parameter W. See the time_warp() method for details
        
        self.W = time_warp_param
        
        # time masking parameter T. See the time_mask() method for details
        
        self.T = time_mask_param
        
        # frequency masking parameter F. See the frequency_mask() method for
        # details
        
        self.F = frequency_mask_param
        
        # the number of masks to be applied to the time axis of the Mel-scale
        # spectrogram
        
        self.num_time_masks = num_time_masks
        
        # the number of masks to be applied to the frequency axis of the
        # Mel-scale spectrogram
        
        self.num_frequency_masks = num_frequency_masks
        
        """
        if this is set to true, then the number of time masks that are applied
        will be sampled from a discrete uniform distribution with a support
        of (1,T), while the number of frequency masks that are applied will
        be sampled from a discrete uniform distribution with a support of
        (1,F). The number of time masks and frequency masks that are applied
        will be random for each augmentation.
        
        Otherwise, the number of time masks that are applied will be fixed
        and equal to num_time_masks and the number of frequency masks will be
        fixed and equal to num_frequency_masks.
        """
        
        self.num_masks_random = num_masks_random
        
        # initialize random number generator
        
        self.rng = np.random.default_rng()
    
    def time_warp(self,mel_spec):
    
        """
        Given a Mel-scale spectrogram with shape v x tau, time
        warping is done as follows:
        
        1. Pick a column on the time axis at random from column number W to column
           number tau - W, where W is the time warp parameter that is chosen
           beforehand. In the original paper, W ranges from 0 to 80. Let the chosen
           column number be alpha. This means that the Mel-spectrogram is now
           partitioned into two windows defined on the time axis as:
               
           1st window:
               column 0 --> column alpha
               
           2nd window:
               column alpha + 1 --> column tau
        
        2. The 1st window is to be warped to the left or right by a distance w that
           is sampled from a discrete uniform distribution with a support of [-W,W].
           The 2nd window is warped in the opposite direction by the same amount
           as the 1st window.
        3. To perform this warping, control points are placed on each pixel in the
           alpha column. The final locations of these pixels after warping are also
           specified. Finally, the Tensorflow Addons function
           tensorflow_addons.image.sparse_image_warp() is then used to perform time
           warping.
        
        The original SpecAugment paper can be found here:
            
        https://arxiv.org/abs/1904.08779
            
        """
        
        # width and height of Mel-scale spectrogram. Height is equivalent to number
        # of Mels and width is equivalent to number of time steps.
        # v and tau are used to match the notation used in the original paper
        
        v, tau = mel_spec.shape
        
        # alpha is the column to be warped
        
        alpha = self.rng.integers(self.W,tau - self.W,
                                  endpoint = True)
    
        # each row of this array will contain the locations of the points to be
        # warped. More precisely, it contains the coordinates of the alpha column,
        # such that each pixel in the alpha column has its own coordinate
    
        src_ctrl_pt_loc = np.zeros((v,2),dtype=np.float32)
        
        # each row of the alpha column
        
        src_ctrl_pt_loc[:,0] = np.arange(v)
        
        # the alpha column
        
        src_ctrl_pt_loc[:,1] = alpha
    
        # add a batch dimension for the sparse_image_warp function to be used later
    
        src_ctrl_pt_loc = np.expand_dims(src_ctrl_pt_loc, 0)
        
        # sample w from a discrete uniform distribution
        
        w = self.rng.integers(-self.W,self.W,
                         endpoint = True)
    
        # each row of this array will contain the locations of the destinations that
        # the points in src_ctrl_pt_loc will be warped to. These coordinates
        # correspond to the target column
    
        dst_ctrl_pt_loc = np.zeros((v,2),dtype=np.float32)
        
        # each row of the target column
        
        dst_ctrl_pt_loc[:,0] = np.arange(v)
        
        # the target column
        
        dst_ctrl_pt_loc[:,1] = alpha + w
    
        # add a batch dimension for the sparse_image_warp function to be used later
    
        dst_ctrl_pt_loc = np.expand_dims(dst_ctrl_pt_loc, 0)
    
        # add batch and channel dimensions for the sparse_image_warp function to be
        # used later
    
        mel_spec = np.reshape(mel_spec,(1,*mel_spec.shape,1))
        
        """    
        the sparse_image_warp function returns a tuple, where the first element
        is the warped Mel-scale spectrogram and the second element is the dense
        flow field that is produced by interpolation. Only the warped Mel-scale
        spectrogram is of interest. Also, because the function returns a
        tensorflow tensor, it is converted to a numpy array using the .numpy()
        method.
        
        the num_boundary_points argument is set to be greater than 0 such that
        frequency content is not lost while warping by constraining the warping
        transformation.
        """
        
        mel_spec = sparse_image_warp(
                            image = mel_spec,
                            source_control_point_locations = src_ctrl_pt_loc,
                            dest_control_point_locations = dst_ctrl_pt_loc,
                            interpolation_order = 2,
                            regularization_weight = 0.0,
                            num_boundary_points = 1)[0].numpy()
        
        # remove redundant dimensions
        
        mel_spec = np.squeeze(mel_spec)
    
    def time_mask(self,mel_spec,num_masks):
        
        # number of time steps in spectrogram
        
        tau = mel_spec.shape[1]
        
        # generate num_masks time masks
        
        for _ in range(0,num_masks):
            
            t = self.rng.integers(0,self.T,
                             endpoint = True)
            
            t_0 = self.rng.integers(0,tau - t,
                               endpoint = True)
            
            mel_spec[:, t_0:t_0 + t] = np.mean(mel_spec)
        
    def frequency_mask(self,mel_spec,num_masks):
        
        # number of Mels in spectrogram
        
        v = mel_spec.shape[0]
        
        # generate num_masks frequency masks
        
        for _ in range(0,num_masks):
            
            f = self.rng.integers(0,self.F,
                             endpoint = True)
            
            f_0 = self.rng.integers(0,v - f,
                               endpoint = True)
            
            mel_spec[f_0:f_0 + f,:] = np.mean(mel_spec)
        
    def __call__(self,mel_spectrogram):
        
        # lists to store augmentations
        
        augmentations = []
        # TODO: remove this
            
        time_warped=[]
        time_masked=[]
        frequency_masked=[]
        # apply random transformations. Note that mel_spec is modified in-place
        # using these methods because it is a mutable numpy array
        
        i=0
        
        for _ in range(self.num_aug):
            
            # numpy arrays are mutable, so need to make copy to avoid modifying
            # original array
            
            mel_spec = mel_spectrogram.copy()
            
            x = self.rng.uniform(low=0.0,high=1.0)
            
            if x <= self.aug_split[0]:
                
                self.time_warp(mel_spec)
                time_warped.append(1)
            
            if x <= self.aug_split[1]:
                
                # if a random number of time masks are to be applied
                
                if self.num_masks_random:
                    num_t_masks = self.rng.integers(1,self.num_time_masks,
                                                    endpoint = True)
                else:
                    num_t_masks = self.num_time_masks
                
                self.time_mask(mel_spec,num_t_masks)
                time_masked.append(1)
            
            if x <= self.aug_split[2]:
                
                # if a random number of frequency masks are to be applied
                
                if self.num_masks_random:
                    num_f_masks = self.rng.integers(1,self.num_frequency_masks,
                                                    endpoint = True)
                else:
                    num_f_masks = self.num_frequency_masks
                
                self.frequency_mask(mel_spec,num_f_masks)
                frequency_masked.append(1)
            
            # TODO: remove this
            
            if (x > self.aug_split[0] and x > self.aug_split[1] and 
                x > self.aug_split[2]):
                
                continue
                
                # x = self.rng.integers(1,3,endpoint=True)
                
                # if x == 1:
                #     self.time_warp(mel_spec)
                #     time_warped.append(1)
                # if x == 2:
                #     # if a random number of time masks are to be applied
                
                #     if self.num_masks_random:
                #         num_t_masks = self.rng.integers(1,self.num_time_masks,
                #                                         endpoint = True)
                #     else:
                #         num_t_masks = self.num_time_masks
                    
                #     self.time_mask(mel_spec,num_t_masks)
                #     time_masked.append(1)
                # if x == 3:
                #     # if a random number of frequency masks are to be applied
                
                #     if self.num_masks_random:
                #         num_f_masks = self.rng.integers(1,self.num_frequency_masks,
                #                                         endpoint = True)
                #     else:
                #         num_f_masks = self.num_frequency_masks
                    
                #     self.frequency_mask(mel_spec,num_f_masks)
                #     frequency_masked.append(1)
            
            # save the augmentation
            
            augmentations.append(mel_spec)
            
        return augmentations,time_warped,time_masked,frequency_masked

if __name__ == '__main__':
    
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    
    x,sr = librosa.load(path = 'test.wav',
                        sr = None,
                        mono = True)

    mel_spec = librosa.feature.melspectrogram(y = x,
                                              sr = sr,
                                              n_fft = 2048,
                                              hop_length = 32,
                                              power = 2.0,
                                              n_mels = 256)
    
    spec_augment = SpecAugment(num_aug = 200,
                               aug_split = [0.1,0.7,0.4],
                               time_warp_param = 30,
                               time_mask_param = 30,
                               frequency_mask_param = 30,
                               num_time_masks = 2,
                               num_frequency_masks = 3,
                               num_masks_random = True)
    
    augmentations,time_warped,time_masked,frequency_masked = spec_augment(mel_spec)
            