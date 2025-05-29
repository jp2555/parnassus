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

'''particles = [pT, eta, phi, PID, z]'''
'''NO JET dateset. These are all particles EIC events'''

PID_INDEX = 9

labels = {
    'small_test_file.h5'
    # 'pythia8NCDIS_18x275_minQ2=100.h5'
}



def process(p, gen_ele):
    '''
    This function should be called for gen and reco particles.
    It's main purpose is to normalize particle features relative to the scattered electron
    And provide PID boolean features for each particle. Note, we do not save charge ATM 5/29/25
     
    A key pre-processing step of mean->0.0 and stdev->1.0 is done in utils.py
    In dis_root_to_h5.py, the leading electron is set as the first particle in the array

    If the first particle is not an electron, that means it was not reconstructed.
    If that is the case, we normalize reco particles by the Gen. electron here
    Vinny's idea is to then train a classifier at gen-level to determine if electron is reconstructed

    gen_ele is Gen. Level electron
    '''

    #make a deep copy of electron eta and P to leave original un-modified
    #we modify these new arrays to be reco or Gen electron
    e_eta = p[:,0,-2].copy()
    e_P = p[:,0,0].copy()

    mask = np.abs(p[:,0,PID_INDEX]) == 11  # mask to see if first particle is electron
    e_eta = np.where(mask, e_eta, gen_ele[:,-2])  # set e_eta to gen_ele_eta if MASK=0
    e_P   = np.where(mask, e_P,   gen_ele[:,0])   # set e_P to gen_ele_P if MASK=0

    #log(P_rel)
    P_rel = np.ma.masked_invalid(np.divide(p[:, 1:, 0], e_P[:, np.newaxis]))
    log_P_rel = np.ma.log(P_rel)
    log_P_rel = log_P_rel.filled(0)

    #new particles array excludes first particle (electron)
    new_p = np.zeros(shape=(p.shape[0],p.shape[1]-1,11))
    
    new_p[:,:,0] = p[:,1:,-2] + e_eta[:, np.newaxis] # eta rel
    new_p[:,:,1] = p[:,1:,-1]         # phi
    new_p[:,:,2] = log_P_rel
    new_p[:,:,3] = p[:,1:,4]         # x-vert.
    new_p[:,:,4] = p[:,1:,5]         # y-vert.
    new_p[:,:,5] = p[:,1:,6]         # z-vert.nt 


    #hardcoded pid booleans
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

    new_p[:,:,6] = np.abs(p[:,1:,PID_INDEX]) == 11. #electron
    new_p[:,:,7] = np.abs(p[:,1:,PID_INDEX]) == 13. #muon
    new_p[:,:,8] = np.abs(p[:,1:,PID_INDEX]) == 22. #photon
    new_p[:,:,9] = np.isin(np.abs(p[:,1:,PID_INDEX]).astype(int), neutral_hads)  # is neutral hadron
    new_p[:,:,10] = np.isin(np.abs(p[:,1:,PID_INDEX]).astype(int), charged_hads) # is charged hadron


    ic(new_p[1:3])
    new_p = new_p*mask[:,np.newaxis,np.newaxis]
    print("\n\nLine 84, after MASK\n")
    ic(new_p[1,:3])
    # print(new_p[:2])


    #Now Edit electron, before returning
    e_phi = np.where(mask, p[:,0,-1], gen_ele[:,-1])
    e_vx   = np.where(mask, p[:,0,4],   gen_ele[:,4])
    e_vy   = np.where(mask, p[:,0,5],   gen_ele[:,5])
    e_vz   = np.where(mask, p[:,0,6],   gen_ele[:,6])
    e_is_el   = np.ones(len(gen_ele))
    e_is_mu   = np.zeros(len(gen_ele))
    e_is_pho   = np.zeros(len(gen_ele))
    e_is_nhad   = np.zeros(len(gen_ele))  # is neutral_hadron
    e_is_chad   = np.zeros(len(gen_ele))

    features = [
        e_eta, e_phi, e_P,
        e_vx, e_vy, e_vz,
        e_is_el, e_is_mu, e_is_pho,
        e_is_nhad, e_is_chad
    ]
    electrons = np.stack(features, axis=1)

    return new_p, electrons


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

            gen_particles = h5f['gen_particle_features'][:nevent_max,:npart_max].astype(np.float32)
            gen_events = h5f['gen_InclusiveKinematicsTruth'].astype(np.float32)

            gen_electrons = gen_particles[:,0,:]
            ic(np.shape(gen_particles))
            ic(np.shape(gen_electrons))

            gen_particles, gen_electrons = process(gen_particles, gen_electrons)

            ic(np.shape(gen_particles))
            ic(np.shape(gen_electrons))


            reco_particles = h5f['reco_particle_features'][:nevent_max,:npart_max].astype(np.float32)
            reco_events = h5f['reco_InclusiveKinematicsESigma'].astype(np.float32)

            ic(np.shape(reco_particles))
            reco_particles, reco_electrons = process(reco_particles, gen_electrons)
            ic(np.shape(reco_particles))


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

            # train['gen_particles'].append(gen_particles[:int(0.63*ntotal)])
            # train['reco_particles'].append(reco_particles[:int(0.63*ntotal)])
            # train['gen_events'].append(gen_events[:int(0.63*ntotal)])
            # train['reco_events'].append(reco_events[:int(0.63*ntotal)])

            # val['gen_particles'].append(gen_particles[int(0.63*ntotal):int(0.7*ntotal)])
            # val['reco_particles'].append(reco_particles[int(0.63*ntotal):int(0.7*ntotal)])
            # val['gen_events'].append(gen_events[int(0.63*ntotal):int(0.7*ntotal)])
            # val['reco_events'].append(reco_events[int(0.63*ntotal):int(0.7*ntotal)])

            # test['gen_particles'].append(gen_particles[int(0.7*ntotal):])
            # test['reco_particles'].append(reco_particles[int(0.7*ntotal):])
            # test['gen_events'].append(gen_events[int(0.7*ntotal):])
            # test['reco_events'].append(reco_events[int(0.7*ntotal):])


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

    train, val, test = preprocess(os.path.join(flags.folder, 'parnassus_TrainTestVal'),labels, flags.nevent_max, flags.npart_max)

    with h5.File('{}/train_{}.h5'.format(os.path.join(flags.folder, 'parnassus_TrainTestVal'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('gen_particles', data=train['gen_particles'])
        dset = fh5.create_dataset('gen_events', data=train['gen_events'])
        dset = fh5.create_dataset('reco_particles', data=train['reco_particles'])
        dset = fh5.create_dataset('reco_events', data=train['reco_events'])


    with h5.File('{}/test_{}.h5'.format(os.path.join(flags.folder, 'parnassus_TrainTestVal'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('gen_particles', data=test['gen_particles'])
        dset = fh5.create_dataset('gen_events', data=test['gen_events'])
        dset = fh5.create_dataset('reco_particles', data=test['reco_particles'])
        dset = fh5.create_dataset('reco_events', data=test['reco_events'])


    with h5.File('{}/val_{}.h5'.format(os.path.join(flags.folder, 'parnassus_TrainTestVal'),"eic"), "w") as fh5:
        dset = fh5.create_dataset('gen_particles', data=val['gen_particles'])
        dset = fh5.create_dataset('gen_events', data=val['gen_events'])
        dset = fh5.create_dataset('reco_particles', data=val['reco_particles'])
        dset = fh5.create_dataset('reco_events', data=val['reco_events'])
