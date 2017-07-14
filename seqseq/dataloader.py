import collections
import cPickle
import h5py
import math
import numpy as np
import os
import random
import pdb
# Class to load and preprocess data
class DataLoader():
    def __init__(self, batch_size, val_frac, seq_length):
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seq_length = seq_length

        print 'validation fraction: ', self.val_frac

        print "loading data..."
        self._load_data()

        print 'creating splits...'
        self._create_split()

        print 'shifting/scaling data...'
        self._shift_scale()

    def _trim_data(self, full_s, full_a, intervals, data_subsamp_freq=1):
        # Python indexing; find bounds on data given seq_length
        intervals -= 1
        lengths = np.floor(np.diff(np.append(intervals, len(full_s)-1))/self.seq_length)*self.seq_length
        intervals = np.vstack((intervals, intervals + lengths)).T.astype(int)
        ret_bounds = np.insert(np.cumsum(lengths), 0, 0.).astype(int)

        # Remove states that don't fit due to value of seq_length
        s = np.zeros((int(sum(lengths)), full_s.shape[1]))
        for i in xrange(len(ret_bounds)-1):
            s[ret_bounds[i]:ret_bounds[i+1]] = full_s[intervals[i, 0]:intervals[i, 1]]
        s = np.reshape(s, (-1, self.seq_length, full_s.shape[1]))

        # Remove actions that don't fit due to value of seq_length
        a = np.zeros((int(sum(lengths)), full_a.shape[1]))
        for i in xrange(len(ret_bounds)-1):
            a[ret_bounds[i]:ret_bounds[i+1]] = full_a[intervals[i, 0]:intervals[i, 1]]
        a = np.reshape(a, (-1, self.seq_length, full_a.shape[1]))

        return s[:,::data_subsamp_freq], a[:,::data_subsamp_freq]

    def _load_data(self):
        data_dir = '../expert_trajs/'

        # Load mixed data
        filename = data_dir + 'core1_temp0_well1_neig0_carl1_roal0_clrr1_mtl100_clb20_rlb20_rll2_clmr100_rlmr50_seed456.h5'
        expert_data, _ = self.load_trajs(filename, swap=False)
        #s, a = self._trim_data(expert_data['states'], expert_data['actions'], expert_data['exlen_B'])
        
        #data = h5py.File(filename, 'r')
        s = expert_data['states']
        a = expert_data['actions']
        
        # Make sure batch_size divides into num of examples 
        self.s = s[:int(np.floor(len(s)/self.batch_size)*self.batch_size)]
        self.s = np.reshape(self.s, (-1, self.batch_size, self.seq_length, s.shape[2]))
        self.a = a[:int(np.floor(len(a)/self.batch_size)*self.batch_size)]
        self.a = np.reshape(self.a, (-1, self.batch_size, self.seq_length, a.shape[2]))

        # Now separate states and classes
        
        # Print tensor shapes
        print 'states: ', self.s.shape
        print 'actions: ', self.a.shape
        #pdb.set_trace()

        # Create batch_dict
        self.batch_dict = {}
        self.batch_dict["states"] = np.zeros((self.batch_size, self.seq_length, s.shape[2]))
        self.batch_dict["actions"] = np.zeros((self.batch_size, self.seq_length, a.shape[2]))

        # Shuffle data
        print 'shuffling...'
        p = np.random.permutation(len(self.s))
        self.s = self.s[p]
        self.a = self.a[p]

    # Separate data into train/validation sets
    def _create_split(self):

        # compute number of batches
        self.n_batches = len(self.s)
        self.n_batches_val = int(math.floor(self.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print 'num training batches: ', self.n_batches_train
        print 'num validation batches: ', self.n_batches_val

        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Shift and scale data to be zero-mean, unit variance
    def _shift_scale(self):
        # Find means and std, ignore state values that are indicators
        self.shift_s = np.mean(self.s[:self.n_batches_train], axis=(0, 1, 2))
        self.scale_s = np.std(self.s[:self.n_batches_train], axis=(0, 1, 2))
        self.shift_a = np.mean(self.a[:self.n_batches_train], axis=(0, 1, 2))
        self.scale_a = np.std(self.a[:self.n_batches_train], axis=(0, 1, 2))

        # Get rid of scale for indicator features
        self.scale_s = np.array([1.0*(s < 1e-3) + s for s in self.scale_s])

        # Transform data
        self.s = (self.s - self.shift_s)/self.scale_s
        self.a = (self.a - self.shift_a)/self.scale_a
        #pdb.set_trace()

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train]
        self.batch_dict["states"] = self.s[batch_index]
        self.batch_dict["actions"] = self.a[batch_index]

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(self.n_batches_train)
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = self.batchptr_val + self.n_batches_train-1
        self.batch_dict["states"] = self.s[batch_index]
        self.batch_dict["actions"] = self.a[batch_index]

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0

    def load_trajs(self, filename, limit_trajs=None, swap=False):
        # Load expert data
        with h5py.File(filename, 'r') as f:
            # Read data as written by scripts/format_data.py (openai format)
            pdb.set_trace()
            if swap:
                obs = np.array(f['obs_B_T_Do']).T
                act = np.array(f['a_B_T_Da']).T
                #rew= np.array(f['r_B_T']).T
                lng = np.array(f['len_B']).T
            else:
                obs = np.array(f['obs_B_T_Do'])
                act = np.array(f['a_B_T_Da'])
                #rew= np.array(f['r_B_T'])
                lng = np.array(f['len_B'])
    
            full_dset_size = obs.shape[0]
            dset_size = min(
                full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size
    
            exobs_B_T_Do = obs[:dset_size, ...][...]
            exa_B_T_Da = act[:dset_size, ...][...]
            #exr_B_T = rew[:dset_size,...][...]
            exlen_B = lng[:dset_size, ...][...]
    
        # compute trajectory intervals from lengths.
        interval = np.ones(full_dset_size,).astype(int)
        for i, l in enumerate(exlen_B):
            if i == 0:
                continue
            interval[i] = interval[i - 1] + l
    
        stats = {'N': dset_size}
        # record trajectory statistics
        stats['obs_min'] = np.nanmin(exobs_B_T_Do, axis=(0, 1))
        stats['obs_max'] = np.nanmax(exobs_B_T_Do, axis=(0, 1))
    
        stats['obs_minsq'] = np.nanmin(exobs_B_T_Do ** 2., axis=(0, 1))
        stats['obs_maxsq'] = np.nanmax(exobs_B_T_Do ** 2., axis=(0, 1))
    
        stats['obs_mean'] = np.nanmean(exobs_B_T_Do, axis=(0, 1))
        stats['obs_meansq'] = np.nanmean(np.square(exobs_B_T_Do), axis=(0, 1))
        stats['obs_std'] = np.nanstd(exobs_B_T_Do, axis=(0, 1))
    
        stats['act_mean'] = np.nanmean(exa_B_T_Da, axis=(0, 1))
        stats['act_meansq'] = np.nanmean(np.square(exa_B_T_Da), axis=(0, 1))
        stats['act_std'] = np.nanstd(exa_B_T_Da, axis=(0, 1))
    
        data = {'states': exobs_B_T_Do,
                'actions': exa_B_T_Da,
                #'exr_B_T' : exr_B_T,
                'exlen_B': exlen_B,
                'interval': interval
                }
        return data, stats
       
    def prepare_trajs(self, exobs_B_T_Do, exa_B_T_Da, exlen_B, data_subsamp_freq=1, labeller=None):
        print('exlen_B inside: %i' % exlen_B.shape[0])
    
        start_times_B = np.random.RandomState(0).randint(
            0, data_subsamp_freq, size=exlen_B.shape[0])
        exobs_Bstacked_Do = np.concatenate(
            [exobs_B_T_Do[i, start_times_B[i]:l:data_subsamp_freq, :]
                for i, l in enumerate(exlen_B)],
            axis=0)
        exa_Bstacked_Da = np.concatenate(
            [exa_B_T_Da[i, start_times_B[i]:l:data_subsamp_freq, :]
                for i, l in enumerate(exlen_B)],
            axis=0)
    
        assert exobs_Bstacked_Do.shape[0] == exa_Bstacked_Da.shape[0]
    
        data = {'states': exobs_Bstacked_Do,
                'actions': exa_Bstacked_Da}
    
        return data
