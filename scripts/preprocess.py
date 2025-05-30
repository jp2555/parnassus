import h5py as h5
import os
import sys
import numpy as np
from optparse import OptionParser
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from sklearn.utils import shuffle

from icecream import ic
ic.configureOutput(includeContext=True)


PID_INDEX = 9

labels = { #file names
    # 'small_test_file.h5'
    'pythia8NCDIS_18x275_minQ2=100.h5'
}

# Hardcoded pid booleans
neutral_hads = [
    111,   # pi0
    2112,  # neutron
    130,   # K0L
    310,   # K0S
    311,   # K0
    3122,  # Lambda0
]

charged_hads = [
    211,    # pi±
    321,    # K±
    2212    # p / p̄
]  


def process(p, gen_ele):
    '''
    This function should be called for gen and reco particles.
    It's main purpose is to normalize particle features relative to the scattered electron
    And provide PID boolean features for each particle. Note, we do not save charge ATM 5/29/25
     
    A key pre-processing step of mean->0.0 and stdev->1.0 is done in utils.py

    In dis_root_to_h5.py, the leading electron is set as the first particle in the array
    If the first reco-particle is not an electron, that means one was not reconstructed.
    If that is the case, we normalize reco particles by the Gen. electron here (gen_ele)
    Vinny's idea is to then train a classifier at gen-level to determine if electron is reconstructed
    '''

    # Return Feature Dimensions
    N_Ele_Feat = 6     # eta, phi, E, xyz-vert
    N_Part_Feat = 11   # ^ + five PID booleans

    electrons = p[:,0,:]
    mask = np.abs(electrons[:,PID_INDEX]) == 11   # mask if first particle is actually electron
    ic(np.shape(mask[:,np.newaxis]))
    ic(np.shape(electrons))
    ic(np.shape(gen_ele))
    ic(np.shape(p))
    electrons = np.where(mask[:, np.newaxis], electrons, gen_ele)


    # Get log(P_rel)
    e_P = electrons[:,0]
    P_rel = np.ma.masked_invalid(np.divide(p[:, 1:, 0], e_P[:, np.newaxis]))
    log_P_rel = np.ma.log(P_rel)
    log_P_rel = log_P_rel.filled(0)

    #new particles array excludes first particle (electron)
    new_p = np.zeros(shape=(p.shape[0],p.shape[1]-1, N_Part_Feat))
    
    # new_p[:,:,0] = p[:,1:,-2] + e_eta[:, np.newaxis] # eta rel
    new_p[:,:,0] = p[:,1:,-2] + electrons[:,-2][:, np.newaxis]      # eta rel
    new_p[:,:,1] = p[:,1:,-1]                                       # phi
    new_p[:,:,2] = log_P_rel                                        # log(P_rel)
    new_p[:,:,3] = p[:,1:,4]  + electrons[:,4][:, np.newaxis]       # x-vert.
    new_p[:,:,4] = p[:,1:,5]  + electrons[:,5][:, np.newaxis]       # y-vert.
    new_p[:,:,5] = p[:,1:,6]  + electrons[:,6][:, np.newaxis]       # z-vert.

    new_p[:,:,6] = np.abs(p[:,1:,PID_INDEX]) == 11.                 # is electron
    new_p[:,:,7] = np.abs(p[:,1:,PID_INDEX]) == 13.                 # is muon
    new_p[:,:,8] = np.abs(p[:,1:,PID_INDEX]) == 22.                 # is photon
    new_p[:,:,9] = np.isin(np.abs(p[:,1:,PID_INDEX]).astype(int), neutral_hads)  # is neutral hadron
    new_p[:,:,10] = np.isin(np.abs(p[:,1:,PID_INDEX]).astype(int), charged_hads) # is charged hadron


    # ic(new_p[1:3])
    new_p = new_p*mask[:,np.newaxis,np.newaxis]
    print("\n\nLine 84, after MASK\n")
    # ic(new_p[1,:3])


    #Now Edit electron, before returning
    new_ele = np.zeros(shape=(electrons.shape[0], N_Ele_Feat)) #[Nevt, Nfeat]
    new_ele[:,0] = electrons[:,-2]
    new_ele[:,1] = electrons[:,-1]
    new_ele[:,2] = electrons[:, 0]
    new_ele[:,3] = electrons[:, 4]
    new_ele[:,4] = electrons[:, 5]
    new_ele[:,5] = electrons[:, 6]

    return new_p, new_ele


def preprocess(path, labels, nevent_max=-1, npart_max=-1):

    train = {
        'gen_events':[],
        'gen_particles':[],
        'reco_events':[],
        'reco_particles':[],
    }

    test = {
        'gen_events':[],
        'gen_particles':[],
        'reco_events':[],
        'reco_particles':[],
    }

    val = {
        'gen_events':[],
        'gen_particles':[],
        'reco_events':[],
        'reco_particles':[],
    }

    for label in labels:

        with h5.File(os.path.join(path,label),"r") as h5f:

            ntotal = h5f['reco_particle_features'][:nevent_max].shape[0]

            reco_particles = h5f['reco_particle_features'][:nevent_max,:npart_max].astype(np.float32)
            reco_events = h5f['reco_InclusiveKinematicsESigma'].astype(np.float32)

            gen_particles = h5f['gen_particle_features'][:nevent_max,:npart_max].astype(np.float32)
            gen_events = h5f['gen_InclusiveKinematicsTruth'].astype(np.float32)

            gen_electrons = gen_particles[:,0,:]

            ic("Before Process", np.shape(reco_particles))
            reco_particles, reco_electrons = process(reco_particles, gen_electrons)
            ic("After Process", np.shape(reco_particles))


            ic("Before Process", np.shape(gen_particles))
            gen_particles, gen_electrons = process(gen_particles, gen_electrons)
            ic("After Process", np.shape(gen_particles))

            ic(np.shape(gen_events))
            gen_events = np.hstack((gen_events, gen_electrons))
            ic(np.shape(gen_events))
            reco_events = np.hstack((reco_events, reco_electrons))

            # Train, Val, test Split
            datasets = {'train': train, 'val': val, 'test': test}
            splits = { 'train': (0, 0.63), 'val':   (0.63, 0.7), 'test':  (0.7, 1.0) }

            for split_name, (start_frac, end_frac) in splits.items():
                start = int(start_frac * ntotal)
                end = int(end_frac * ntotal)

                for key, data in zip(
                    ['gen_particles', 'reco_particles', 'gen_events', 'reco_events'],
                    [gen_particles, reco_particles, gen_events, reco_events]
                ):
                    datasets[split_name][key].append(data[start:end])
            # FIXME: 5/29: make sure we don't just pull the 200GB dataset into memory all at once...


    for key in train:        
        train[key] = np.concatenate(train[key],0)        
    for key in test:
        test[key] = np.concatenate(test[key],0)
    for key in val:
        val[key] = np.concatenate(val[key],0)


    for d in [train, val, test]:
        (
            d['gen_particles'],
            d['reco_particles'],
            d['gen_events'],
            d['reco_events']
        ) = shuffle(
            d['gen_particles'],
            d['reco_particles'],
            d['gen_events'],
            d['reco_events'],
            random_state=42  # makes shuffling reproducible
        )

    return train, val, test

if __name__=='__main__':

    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/global/cfs/cdirs/m3246/eic/NC_DIS_18x275/', help="Folder containing input files")
    parser.add_option("--nevent_max", type="int", default='10_000_000', help="max number of events")
    parser.add_option("--npart_max", type="int", default='200', help="max number of particses per event")
    (flags, args) = parser.parse_args()

    train, val, test = preprocess(os.path.join(flags.folder, 'h5_files'),labels, flags.nevent_max, flags.npart_max)

    with h5.File('{}/train_{}.h5'.format(os.path.join(flags.folder, 'h5_files'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('gen_particles', data=train['gen_particles'])
        dset = fh5.create_dataset('gen_events', data=train['gen_events'])
        dset = fh5.create_dataset('reco_particles', data=train['reco_particles'])
        dset = fh5.create_dataset('reco_events', data=train['reco_events'])


    with h5.File('{}/test_{}.h5'.format(os.path.join(flags.folder, 'h5_files'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('gen_particles', data=test['gen_particles'])
        dset = fh5.create_dataset('gen_events', data=test['gen_events'])
        dset = fh5.create_dataset('reco_particles', data=test['reco_particles'])
        dset = fh5.create_dataset('reco_events', data=test['reco_events'])


    with h5.File('{}/val_{}.h5'.format(os.path.join(flags.folder, 'h5_files'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('gen_particles', data=val['gen_particles'])
        dset = fh5.create_dataset('gen_events', data=val['gen_events'])
        dset = fh5.create_dataset('reco_particles', data=val['reco_particles'])
        dset = fh5.create_dataset('reco_events', data=val['reco_events'])
