import h5py as h5
import os
import sys
import numpy as np
from optparse import OptionParser
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from sklearn.utils import shuffle

'''particles = [pT, eta, phi, PID, z]'''
'''NO JET dateset. These are all particles EIC events'''

PID_INDEX = 9

labels = {
    'Pythia_26812400_10100000.h5'
}



def process(p, gen_e):

    #IMPORTANT setting mean=0.0 and stdev=1.0 is done in utils.py

    #First particle is Electron, if in the event (reco sometimes does not have it)
    #Vinny's idea is to normalize by the leading Gen. electron only if there is no reco-leading electron
    #Then, we'd need to train a classifier at gen level to determin if a reco-level electron will show up


    #Call this for gen and reco particles. 

    e_eta = p[:,0,-2].copy()
    e_P = p[:,0,0].copy()

    mask = np.abs(p[:,0,PID_INDEX]) == 11
    e_eta = np.where(mask, e_eta, gen_e[:,-2])
    e_P   = np.where(mask, e_P,   gen_e[:,0])
    #1. Get multiplicity with np.countnonzero
    #2. Save to event array
    #3. Convert PID to charge and simple booleans for easier training

    new_p = np.zeros(shape=(p.shape[0],p.shape[1]-1,11))
    
    new_p[:,:,0] = p[:,1:,-2] + e_eta # eta rel
    new_p[:,:,1] = p[:,1:,-1]                             # phi
    new_p[:, :, 2] = np.ma.log(np.divide(p[:, 1:, 0], e_P).filled(0)).filled(0)  # pT rel

    new_p[:,:,3] = p[:,1:,4]                             # x-vert.
    new_p[:,:,4] = p[:,1:,5]                             # y-vert.
    new_p[:,:,5] = p[:,1:,6]                             # z-vert.nt 

    #hardcoded pid booleans

    neutral_hads = 
    [
        111,   # pi0
        2112,  # neutron
        130,   # K0L
        310,   # K0S
        311,   # K0
        3122,  # Lambda0
    ]

    charged_hads = 
    [
        211,    # pi±
        321,    # K±
        2212    # p / p̄
    ]  

    new_p[:,:,6] = np.abs(p[:,1:,PID_INDEX]) == 11. #electron
    new_p[:,:,7] = np.abs(p[:,1:,PID_INDEX]) == 13. #muon
    new_p[:,:,8] = np.abs(p[:,1:,PID_INDEX]) == 22. #photon
    new_p[:,:,9] = np.isin(np.abs(p[:,1:,PID_INDEX]).astype(int), neutral_hads)  # is neutral hadron
    new_p[:,:,10] = np.isin(np.abs(p[:,1:,PID_INDEX]).astype(int), charged_hads) # is charged hadron


    print("\n\nLine 60, AFTER feature shuffle\n", new_p[1,:3])
    # the main DataLoader class calculates disttanc/s, expecting eta and phi at 0 and 1

    new_p = new_p*mask[:,:,None]
    print("\n\nLine 84, after MASK\n", new_p[1,:3])
    # print(new_p[:2])


    #Now EDit electron, before returning
    e_phi = np.where(mask, p[:,0,-1], gen_e[:,-1])
    e_vx   = np.where(mask, p[:,0,4],   gen_e[:,4])
    e_vy   = np.where(mask, p[:,0,5],   gen_e[:,5])
    e_vz   = np.where(mask, p[:,0,6],   gen_e[:,6])
    e_is_el   = np.ones(len(gen_e))
    e_is_mu   = np.zeros(len(gen_e))
    e_is_pho   = np.zeros(len(gen_e))
    e_is_nhad   = np.zeros(len(gen_e))
    e_is_chad   = np.zeros(len(gen_e))

    features = [
        e_eta,e_phi,e_P,
        e_vx, e_vy, e_vz,
        e_is_el, e_is_mu, e_is_pho,
        e_is_nhad,e_is_chad
    ]
    electrons = np.stack(features, axis=1)

    return new_p, electrons

    def preprocess(path, labels, nevent_max=-1, npart_max=-1):

        train = {
            'gen_event_kin':[],
            'gen_particles':[],
            'reco_event_kin':[],
            'reco_particles':[],
        }

        test = {
            'gen_event_kin':[],
            'gen_particles':[],
            'reco_event_kin':[],
            'reco_particles':[],
        }

        val = {
            'gen_event_kin':[],
            'gen_particles':[],
            'reco_event_kin':[],
            'reco_particles':[],
        }

    for label in labels:        

        with h5.File(os.path.join(path,label),"r") as h5f:

            ntotal = h5f['reco_particle_features'][:nevent_max].shape[0]

            p = h5f['reco_particle_features'][:nevent_max,:npart_max].astype(np.float32)
            e = h5f['gen_InclusiveKinematicsTruth'].astype(np.float32)
            print(np.shape(p))

            #process particles and event information
            p, e = process(p,e)  # applies mask, saves real pid, shuffles feature indecies for training

            train['data'].append(p[:int(0.63*ntotal)])
            train['evt'].append(evt[:int(0.63*ntotal)])

            val['data'].append(p[int(0.63*ntotal):int(0.7*ntotal)])
            val['evt'].append(evt[int(0.63*ntotal):int(0.7*ntotal)])

            test['data'].append(p[int(0.7*ntotal):])
            test['evt'].append(evt[int(0.7*ntotal):])

            gen_p = h5f['gen_particle_features'][:nevent_max,:npart_max].astype(np.float32)
            p,e = process(p)  # saves real pid, shuffles feature indecies for training



    for key in train:        
        train[key] = np.concatenate(train[key],0)        
    for key in test:
        test[key] = np.concatenate(test[key],0)
    for key in val:
        val[key] = np.concatenate(val[key],0)


    for d in [train,test,val]:
    d['data'],d['evt'] = shuffle(d['data'],d['evt'])

    return train, val, test

    if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default='/global/cfs/cdirs/m4662', help="Folder containing input files")
    parser.add_option("--nevent_max", type="int", default='10_000_000', help="max number of events")
    parser.add_option("--npart_max", type="int", default='50', help="max number of particses per event")
    (flags, args) = parser.parse_args()


    train, val, test = preprocess(os.path.join(flags.folder, 'parnassus'),labels, flags.nevent_max, flags.npart_max)

    with h5.File('{}/train_{}.h5'.format(os.path.join(flags.folder, 'parnassus'),"eic"), "w") as fh5:
    dset = fh5.create_dataset('data', data=train['data'])
    dset = fh5.create_dataset('evt', data=train['evt'])


    with h5.File('{}/test_{}.h5'.format(os.path.join(flags.folder, 'parnassus'),"eic"), "w") as fh5:
    dset = fh5.create_dataset('data', data=test['data'])
    dset = fh5.create_dataset('evt', data=test['evt'])


    with h5.File('{}/val_{}.h5'.format(os.path.join(flags.folder, 'parnassus'),"eic"), "w") as fh5:
    dset = fh5.create_dataset('data', data=val['data'])
    dset = fh5.create_dataset('pid', data=val['pid'])
    dset = fh5.create_dataset('evt', data=val['evt'])

