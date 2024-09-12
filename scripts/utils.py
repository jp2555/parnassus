import numpy as np
import h5py as h5
from sklearn.utils import shuffle
import sys
import os
import tensorflow as tf
import gc
import random
import itertools
import pickle, copy
from scipy.stats import norm
import horovod.tensorflow.keras as hvd

def setup_gpus():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

def load_pickle(folder,f):
    file_name = os.path.join(folder,'histories',f.replace(".weights.h5",".pkl"))
    with open(file_name, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict

def revert_npart(npart, name='30'):
    # Reverse the preprocessing to recover the particle multiplicity
    stats = {'608': (153.2334258, 48.13839615)}
    mean, std = stats[name]
    return np.round(npart * std + mean).astype(np.int32)


class DataLoader:
    """Base class for all data loaders with common preprocessing methods."""
    def __init__(self, path, batch_size=512, rank=0, size=1, **kwargs):

        self.path = path
        self.batch_size = batch_size
        self.rank = rank
        self.size = size

        self.mean_part = np.array([0.0, 0.0, -0.00172231,
                                   0.03836162 , 0.06146408 , 0.06212494,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.std_part = np.array([1.0 , 1.0 , 0.01279741,
                                  0.57357428, 0.58680139, 2.94017443,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ])

        self.mean_evt =  np.array([1.53233426e+02, -9.03940100e-01, -5.47091213e-01,  1.42402418e+03])
        self.std_evt  = np.array([48.24956703,  45.25099801,  43.78186693, 181.1162556])

        self.part_names = ['$\eta$', '$\phi$', 'log($1 + p_{Trel}$)',
                           'vx', 'vy', 'vz',
                           'is electron','is muon', 'is photon',
                           'is neutral hadron','is charged hadron']
        self.evt_names = ['Multiplicity','$MET_x$ [GeV]','$MET_y$ [GeV]','HT [GeV]']

        self.load_data(path, batch_size,rank,size)
        self.num_feat = self.reco.shape[2]
        if rank ==0:
            print(f"Loaded dataset with {self.num_part} particles and {self.num_feat} features")

        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
        
    def get_preprocess_cond(self,nevts=-1):
        if nevts<0:
            nevts = self.nevts
        gen = self.preprocess(self.gen,self.gen_mask).astype(np.float32)[:nevts]
        gen_evt = self.preprocess_evt(self.gen_evt).astype(np.float32)[:nevts]
        return gen, self.gen_mask.astype(np.float32)[:nevts], gen_evt

        
    def data_from_file(self,file_path, nevts = None,preprocess=False):
        with h5.File(file_path, 'r') as file:
            reco_data_chunk = file['reco'][:nevts]
            reco_mask_chunk = reco_data_chunk[:, :, 2] != 0
            gen_data_chunk = file['gen'][:nevts]
            gen_mask_chunk = gen_data_chunk[:, :, 2] != 0
            
            
            gen_evt_chunk = file['gen_evt'][:nevts]
            reco_evt_chunk = file['reco_evt'][:nevts]

            if preprocess:
                reco_data_chunk = self.preprocess(reco_data_chunk, reco_mask_chunk)
                gen_data_chunk = self.preprocess(gen_data_chunk, gen_mask_chunk)
                reco_evt_chunk = self.preprocess_evt(reco_evt_chunk)
                gen_evt_chunk = self.preprocess_evt(gen_evt_chunk)
                            
        return [reco_data_chunk,gen_data_chunk,
                reco_mask_chunk,gen_mask_chunk,
                reco_evt_chunk, gen_evt_chunk]


    def make_tfdata(self):
        reco = self.preprocess(self.reco,self.reco_mask).astype(np.float32)
        gen = self.preprocess(self.gen,self.gen_mask).astype(np.float32)
        reco_evt = self.preprocess_evt(self.reco_evt).astype(np.float32)
        gen_evt = self.preprocess_evt(self.gen_evt).astype(np.float32)

        tf_zip = tf.data.Dataset.from_tensor_slices(
            {'input_reco':reco,
             'input_gen':gen,
             'input_reco_mask':self.reco_mask.astype(np.float32),
             'input_gen_mask':self.gen_mask.astype(np.float32),
             'input_reco_evt':reco_evt,
             'input_gen_evt':gen_evt,
             })
        
        del self.reco, self.gen, self.reco_mask, self.gen_mask
        gc.collect()
        
        return tf_zip.cache().shuffle(self.batch_size*100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def load_data(self,path, batch_size=512,rank=0,size=1,nevts=None):
        # self.path = path
        self.reco = h5.File(self.path,'r')['reco'][rank:nevts:size]
        self.gen = h5.File(self.path,'r')['gen'][rank:nevts:size]
        self.reco_evt = h5.File(self.path,'r')['reco_evt'][rank:nevts:size]
        self.gen_evt = h5.File(self.path,'r')['gen_evt'][rank:nevts:size]
        
        self.reco_mask = self.reco[:,:,2] != 0
        self.gen_mask = self.gen[:,:,2] != 0
        
            
        # self.batch_size = batch_size
        self.nevts = h5.File(self.path,'r')['reco_evt'].shape[0] if nevts is None else nevts
        self.num_part = self.reco.shape[1]
        self.num_evt = self.reco_evt.shape[1]


    def preprocess(self,x,mask):                
        num_feat = self.mean_part.shape[0]
        new_features = x.copy()
        new_features[:,:,:num_feat] =  mask[:,:, None]*(x[:,:,:num_feat]-self.mean_part)/self.std_part
        new_features[np.isnan(new_features)] = 0.0
        new_features[np.isinf(new_features)] = 0.0
        return new_features

    def preprocess_evt(self,x):
        new_features = (x-self.mean_evt)/self.std_evt
        new_features[np.isnan(new_features)] = 0.0
        new_features[np.isinf(new_features)] = 0.0
        
        return new_features

    def revert_preprocess(self,x,mask):                
        num_feat = self.mean_part.shape[-1]        
        new_part = mask[:,:, None]*(x[:,:,:num_feat]*self.std_part + self.mean_part)
        #charge
        new_part[:,:,6] = np.sign(new_part[:,:,6])
        #pids
        max_indices = np.argmax(new_part[:,:,7:], axis=-1)
        pids = np.zeros_like(new_part[:,:,7:])
        pids[np.arange(new_part.shape[0])[:, None], np.arange(new_part.shape[1]), max_indices] = 1
        new_part[:,:,7:] = pids
        #zero vertex and pids for neutral particles
        neutral_mask = (pids[:,:,2]==1) | (pids[:,:,3]==1)
        new_part[:,:,3:7] *= (1.0-neutral_mask[:,:,None])
                
        return  new_part

    def revert_preprocess_evt(self,x):
        new_x = self.std_evt*x+self.mean_evt
        #Convert multiplicity back into integers
        new_x[:,0] = np.round(new_x[:,0])
        new_x[:,0] = np.clip(new_x[:,0],1,self.num_part)
        return new_x



