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


    # ic("after mask", new_p[1:3])
    new_p = new_p*mask[:,np.newaxis,np.newaxis]
    # ic("before mask", new_p[1,:3])


    #Now Edit electron, return from here, append to event_data
    new_ele = np.zeros(shape=(electrons.shape[0], N_Ele_Feat)) #[Nevt, Nfeat]
    new_ele[:,0] = electrons[:,-2]
    new_ele[:,1] = electrons[:,-1]
    new_ele[:,2] = electrons[:, 0]
    new_ele[:,3] = electrons[:, 4]
    new_ele[:,4] = electrons[:, 5]
    new_ele[:,5] = electrons[:, 6]

    return new_p, new_ele

def preprocess_in_chunks(path, labels,
                         chunk_size=10_000,
                         nevent_max=-1, npart_max=-1):
    """
    Read each input file in “chunks” of size chunk_size, process those events,
    split into train/val/test, and append directly to three output HDF5s.
    """
    # 1) Prepare three output files with extendable datasets
    train_out = h5.File(os.path.join(path, 'train_eic.h5'), 'w')
    val_out   = h5.File(os.path.join(path, 'val_eic.h5'),   'w')
    test_out  = h5.File(os.path.join(path, 'test_eic.h5'),  'w')

    # create these datasets on first chunk; then we will resize/append.
    outputs = {
        'train': {'file': train_out, 'dsets': {}},
        'val':   {'file': val_out,   'dsets': {}},
        'test':  {'file': test_out,  'dsets': {}}
    }

    # Split fractions:
    splits = {
        'train': (0.00, 0.63),
        'val':   (0.63, 0.70),
        'test':  (0.70, 1.00)
    }

    for label in labels:
        infile = h5.File(os.path.join(path, label), 'r')
        # Determine how many total events are in this file (ignore nevent_max for now):
        ntotal_in_file = infile['reco_particle_features'].shape[0]
        if nevent_max > 0:
            ntotal = min(nevent_max, ntotal_in_file)
        else:
            ntotal = ntotal_in_file

        # Loop over chunks of events [i : i+chunk_size]
        for start in range(0, ntotal, chunk_size):
            end = min(start + chunk_size, ntotal)

            # 2) Read just this slice from disk (no “[:]” of the entire dataset)
            reco_particles_chunk = infile['reco_particle_features'][start:end, :npart_max].astype(np.float32)
            reco_events_chunk    = infile['reco_InclusiveKinematicsESigma'][start:end].astype(np.float32)

            gen_particles_chunk  = infile['gen_particle_features'][start:end, :npart_max].astype(np.float32)
            gen_events_chunk     = infile['gen_InclusiveKinematicsTruth'][start:end].astype(np.float32)

            # Extract the “gen electron” column to pass to process():
            gen_electrons_chunk = gen_particles_chunk[:, 0, :].copy()

            ic("Before Process (chunk)", np.shape(reco_particles_chunk))
            reco_particles_proc, reco_electrons_proc = process(reco_particles_chunk, gen_electrons_chunk)
            ic("After Process (chunk)", np.shape(reco_particles_proc))

            ic("Before Process (gen chunk)", np.shape(gen_particles_chunk))
            gen_particles_proc, gen_electrons_proc = process(gen_particles_chunk, gen_electrons_chunk)
            ic("After Process (gen chunk)", np.shape(gen_particles_proc))

            # Append electron features onto “events” arrays:
            gen_events_proc  = np.hstack((gen_events_chunk,  gen_electrons_proc))
            reco_events_proc = np.hstack((reco_events_chunk, reco_electrons_proc))

            # 3) Now split this chunk into train/val/test by index within the chunk
            nevents_in_chunk = end - start
            for split_name, (f0, f1) in splits.items():
                i0 = int(f0 * nevents_in_chunk)
                i1 = int(f1 * nevents_in_chunk)

                gen_p_split  = gen_particles_proc[i0:i1]
                reco_p_split = reco_particles_proc[i0:i1]
                gen_e_split  = gen_events_proc[i0:i1]
                reco_e_split = reco_events_proc[i0:i1]

                # 4) Append four arrays into the corresponding output HDF5's datasets:
                out = outputs[split_name]
                hf = out['file']

                # On the first chunk, create the datasets if they don’t yet exist.
                if not out['dsets']:
                    # Create four extendable datasets with maxshape=(None, ...)
                    #   We know the number of features/dims, so we set shape=(0, dims…)
                    #   and allow maxshape on axis=0 to be None.
                    out['dsets']['gen_particles']  = hf.create_dataset(
                        'gen_particles',
                        shape=(0,) + gen_p_split.shape[1:],
                        maxshape=(None,) + gen_p_split.shape[1:],
                        dtype=np.float32
                    )
                    out['dsets']['reco_particles'] = hf.create_dataset(
                        'reco_particles',
                        shape=(0,) + reco_p_split.shape[1:],
                        maxshape=(None,) + reco_p_split.shape[1:],
                        dtype=np.float32
                    )
                    out['dsets']['gen_events']     = hf.create_dataset(
                        'gen_events',
                        shape=(0,) + gen_e_split.shape[1:],
                        maxshape=(None,) + gen_e_split.shape[1:],
                        dtype=np.float32
                    )
                    out['dsets']['reco_events']    = hf.create_dataset(
                        'reco_events',
                        shape=(0,) + reco_e_split.shape[1:],
                        maxshape=(None,) + reco_e_split.shape[1:],
                        dtype=np.float32
                    )

                # Helper to append a NumPy array to an existing extendable dataset:
                def append_to_dset(dset, array):
                    old_n = dset.shape[0]
                    new_n = old_n + array.shape[0]
                    dset.resize((new_n,) + dset.shape[1:])
                    dset[old_n:new_n, ...] = array

                # Append each split‐chunk:
                append_to_dset(out['dsets']['gen_particles'],  gen_p_split)
                append_to_dset(out['dsets']['reco_particles'], reco_p_split)
                append_to_dset(out['dsets']['gen_events'],     gen_e_split)
                append_to_dset(out['dsets']['reco_events'],    reco_e_split)

        infile.close()

    # Close all three output files
    for split_name in outputs:
        outputs[split_name]['file'].close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Chunked preprocess")
    parser.add_argument("--folder", type=str,
                        default="/global/cfs/cdirs/m3246/eic/NC_DIS_18x275/",
                        help="Folder containing input files")
    parser.add_argument("--chunk_size", type=int, default=10_000,
                        help="How many events to read/process per chunk")
    parser.add_argument("--nevent_max", type=int, default=-1,
                        help="Maximum number of events (per file) to read; -1 means all")
    parser.add_argument("--npart_max", type=int, default=200,
                        help="Maximum number of particles per event")
    args = parser.parse_args()

    preprocess_in_chunks(
        os.path.join(args.folder, "h5_files"),
        labels,
        chunk_size=args.chunk_size,
        nevent_max=args.nevent_max,
        npart_max=args.npart_max
    )
