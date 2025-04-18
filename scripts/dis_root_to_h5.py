import os
import gc
import fnmatch
import argparse
import logging

import numpy as np
np.str = str
np.object = object

import uproot3 as uproot
import h5py

# Define global lists for keys and features
kinematics_list = ['x', 'Q2', 'W', 'y', 'nu']

reco_particle_list = [
    'ReconstructedParticles.type',
    'ReconstructedParticles.energy',  # energy used for sorting (index 1)
    'ReconstructedParticles.momentum.x',
    'ReconstructedParticles.momentum.y',
    'ReconstructedParticles.momentum.z',
    'ReconstructedParticles.referencePoint.x',
    'ReconstructedParticles.referencePoint.y',
    'ReconstructedParticles.referencePoint.z',
    'ReconstructedParticles.charge',
    'ReconstructedParticles.mass',
    'ReconstructedParticles.PDG'
]

gen_particle_list = [
    'MCParticles.PDG',
    'MCParticles.generatorStatus',  # generatorStatus used for sorting (index 1)
    'MCParticles.simulatorStatus',
    'MCParticles.charge',
    'MCParticles.time',
    'MCParticles.mass',
    'MCParticles.vertex.x',
    'MCParticles.vertex.y',
    'MCParticles.vertex.z',
    'MCParticles.momentum.x',
    'MCParticles.momentum.y',
    'MCParticles.momentum.z'
]


def find_files_with_string(directory, pattern):
    """Find all filenames in a directory matching the pattern."""
    matching_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, pattern):
            matching_files.append(filename)
    return matching_files


def append_to_dataset(h5f, dset_name, data, fixed_chunk_size=2000):
    """
    Append data along axis 0 to a dataset in the HDF5 file.
    If the dataset does not exist, create it as an extendable, chunked dataset using a fixed chunk shape.
    If it exists but is not chunked, delete and re-create it.
    """
    if dset_name in h5f:
        dset = h5f[dset_name]
        if dset.chunks is None:
            print(f"Dataset {dset_name} exists and is not chunked. Deleting it for re-creation.")
            del h5f[dset_name]
    
    if dset_name not in h5f:
        # Use a fixed chunk size for the first dimension.
        chunk_dim = fixed_chunk_size if data.shape[0] > fixed_chunk_size else data.shape[0]
        chunk_shape = (chunk_dim,) + data.shape[1:]
        dset = h5f.create_dataset(
            dset_name,
            data=data,
            maxshape=(None,) + data.shape[1:],
            chunks=chunk_shape
        )
    else:
        dset = h5f[dset_name]
        old_shape = dset.shape
        new_shape = (old_shape[0] + data.shape[0],) + old_shape[1:]
        dset.resize(new_shape)
        dset[old_shape[0]:] = data

def process_chunk(tmp_file, start, end, max_part, max_nonzero):
    """
    Process a chunk of events [start:end] from the open root tree.
    Returns two dictionaries (one for reco, one for gen) where each key holds an array
    for that set of events.
    """
    # Initialize dictionaries to accumulate arrays for the current chunk
    reco_dict = {
        'particle_features': [],
        'InclusiveKinematicsDA': [],
        'InclusiveKinematicsElectron': [],
        'InclusiveKinematicsESigma': [],
        'InclusiveKinematicsJB': [],
        'InclusiveKinematicsSigma': []
    }
    gen_dict = {
        'particle_features': [],
        'InclusiveKinematicsTruth': []
    }

    # --- Process Reconstructed (reco) Quantities ---
    # Loop over each kinematic quantity dataset (e.g. InclusiveKinematicsDA, etc.)
    for key in reco_dict.keys():
        if key != 'particle_features':
            # For each inclusive kinematics group, load each feature in the kinematics list.
            # .pad(1).fillna(0).regular() ensures the data are valid for saving.
            arrays = [
                tmp_file[f"{key}.{feat}"].array()[start:end]
                    .pad(1).fillna(0).regular().squeeze()
                for feat in kinematics_list
            ]
            arr = np.stack(arrays, axis=-1)  # Shape: (n_events_chunk, 5)
            reco_dict[key].append(arr)

    # Process Reconstructed Particle Features
    parts = [
        tmp_file[feat].array()[start:end].pad(max_part).fillna(0).regular()
        for feat in reco_particle_list
    ]
    # Stack so that shape is (n_events_chunk, max_part, num_features)
    reco_pf = np.stack(parts, axis=-1)
    # Sort the particles by energy (Energy stored at index 1)
    order = np.argsort(-reco_pf[:, :, 1], axis=1)
    reco_pf = np.take_along_axis(reco_pf, order[:, :, None], axis=1)
    reco_pf = reco_pf[:, :max_nonzero, :]
    reco_dict['particle_features'].append(reco_pf)

    # Concatenate lists in reco_dict (only one element per key here)
    for key in reco_dict:
        reco_dict[key] = np.concatenate(reco_dict[key], axis=0)

    # --- Process Generator (gen) Quantities ---
    # Process inclusive kinematics for truth
    arrays = [
        tmp_file[f'InclusiveKinematicsTruth.{feat}'].array()[start:end]
            .regular().squeeze()
        for feat in kinematics_list
    ]
    gen_kin = np.stack(arrays, axis=-1)
    gen_dict['InclusiveKinematicsTruth'].append(gen_kin)

    # Process Generator Particle Features
    # Apply a mask for final-state particles: generatorStatus == 1
    genStatus = tmp_file['MCParticles.generatorStatus'].array()[start:end]
    gen_parts = [
        tmp_file[feat].array()[start:end][genStatus == 1]
            .pad(max_part).fillna(0).regular()
        for feat in gen_particle_list
    ]
    # Stack so that shape is (n_events_chunk, max_part, num_features)
    gen_pf = np.stack(gen_parts, axis=-1)
    # Sort the particles using generatorStatus (stored at index 1)
    order_gen = np.argsort(-gen_pf[:, :, 1], axis=1)
    gen_pf = np.take_along_axis(gen_pf, order_gen[:, :, None], axis=1)
    gen_pf = gen_pf[:, :max_nonzero, :]
    gen_dict['particle_features'].append(gen_pf)

    # Concatenate lists in gen_dict
    for key in gen_dict:
        gen_dict[key] = np.concatenate(gen_dict[key], axis=0)

    return reco_dict, gen_dict


def process_files(file_list, base_path, output_file,
                  chunk_size=2000, max_part=1000, max_nonzero=200,
                  total_nevts=None):
    """
    Process each ROOT file and append the results in chunks of events to one large HDF5 file.
    - chunk_size controls the number of events loaded into memory at once.
    - max_part is the padding length for per-event particle arrays.
    - max_nonzero sets the maximum number of (nonzero) particles to store per event.
    - total_nevts (if provided) limits the number of events processed per file.
    """
    with h5py.File(output_file, 'a') as h5f:
        for f in file_list:
            file_path = os.path.join(base_path, f)
            try:
                tmp_file = uproot.open(file_path)['events']
            except Exception as e:
                print(f"Error loading file {f}: {e}")
                continue

            # Determine how many events to process in this file.
            num_events = tmp_file.numentries
            if total_nevts is not None:
                num_events = min(num_events, total_nevts)

            if num_events != 428:
                print(f"SKIPPING {f} with {num_events} events")
                continue  # FIXME: Stopgap for weird root files
            else:
                print(f"Processing file: {f} with {num_events} events")


            # Process the file in chunks
            for start in range(0, num_events, chunk_size):
                end = min(start + chunk_size, num_events)
                reco_chunk, gen_chunk = process_chunk(tmp_file, start, end,
                                                      max_part=max_part,
                                                      max_nonzero=max_nonzero)
                # Append each key from the processed chunk to the HDF5 datasets.
                # For reconstructed (reco) data, prepend "reco_" to the key names.
                for key, data in reco_chunk.items():
                    dset_name = "reco_" + key
                    append_to_dataset(h5f, dset_name, data)
                # For generator (gen) data, prepend "gen_"
                for key, data in gen_chunk.items():
                    dset_name = "gen_" + key
                    append_to_dataset(h5f, dset_name, data)

                print(f"  Processed events {start} to {end}")
            del tmp_file  # free the ROOT file
            gc.collect()
    print("All files processed and appended to", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-input',
                        default='/global/cfs/cdirs/m3246/eic/NC_DIS_18x275/minQ2=100',
                        help='Folder containing ROOT files (data and MC)')
    parser.add_argument('--data-output',
                        default='/global/cfs/cdirs/m3246/eic/NC_DIS_18x275/h5_files',
                        help='Output folder for the HDF5 file')
    parser.add_argument('--sample',
                        default='NC_DIS_Q2_100',
                        help='Sample to process. Options: NC_DIS_Q2_100, NC_DIS_Q2_1000')
    parser.add_argument('--chunk-size', type=int, default=2000,
                        help='Number of events to process per chunk')
    parser.add_argument('--total-nevts', type=int, default=None,
                        help='Total number of events to process per file (optional)')
    args = parser.parse_args()

    if args.sample == 'NC_DIS_Q2_100':
        print("Processing Data (NC_DIS_Q2_100 sample)")
        # Adjust the pattern to your naming scheme.
        file_list = find_files_with_string(args.data_input, 'pythia8NCDIS_18x275_minQ2=100*')
        # Define the output HDF5 file
        output_file = os.path.join(args.data_output, "pythia8NCDIS_18x275_minQ2=100.h5")
        process_files(file_list, args.data_input, output_file,
                      chunk_size=args.chunk_size,
                      total_nevts=args.total_nevts)
