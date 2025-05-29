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

def get_eta_phi(px, py, pz):

    p_mag = np.sqrt(px*px + py*py + pz*pz)
    theta = np.arccos(pz / p_mag)
    eta = -np.log(np.tan(theta/2))

    phi = np.arctan2(py, px)
    return eta, phi

# Define global lists for keys and features
kinematics_list = ['x', 'Q2', 'W', 'y', 'nu']

reco_particle_list = [
    'ReconstructedParticles.energy',  # energy used for sorting (index 1)
    'ReconstructedParticles.momentum.x',
    'ReconstructedParticles.momentum.y',
    'ReconstructedParticles.momentum.z',
    'ReconstructedParticles.referencePoint.x',
    'ReconstructedParticles.referencePoint.y',
    'ReconstructedParticles.referencePoint.z',
    'ReconstructedParticles.charge',
    'ReconstructedParticles.mass',
    'ReconstructedParticles.PDG',
    'ReconstructedParticles.type',
]

gen_particle_list = [
    'MCParticles.generatorStatus',  # generatorStatus (final state particle = 1) used for sorting (index 0). Changed  to E later
    'MCParticles.momentum.x',
    'MCParticles.momentum.y',
    'MCParticles.momentum.z',
    'MCParticles.vertex.x',
    'MCParticles.vertex.y',
    'MCParticles.vertex.z',
    'MCParticles.charge',
    'MCParticles.mass',
    'MCParticles.PDG',
    'MCParticles.time',
    # 'MCParticles.simulatorStatus',
]
    # ETA and PHI are appended to the particle features

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


def load_branch_arrays(tree, branch, feats, start, end,
                       pad: int = None, fill_value: float = None,
                       squeeze: bool = False):
    """
    Load multiple flat arrays from branches like "branch.feat",
    apply optional padding/filling, regularize, squeeze, and stack.
    """
    arrays = []
    for feat in feats:
        arr = tree[f"{branch}.{feat}"].array()[start:end]
        if pad is not None:
            arr = arr.pad(pad)
        if fill_value is not None:
            arr = arr.fillna(fill_value)
        arr = arr.regular()
        if squeeze:
            arr = arr.squeeze()
        arrays.append(arr)
    return np.stack(arrays, axis=-1)  # shape (n_events, len(feats))


def swap_leading_electron(pf: np.ndarray, pdg_idx: int = 9):
    """
    Find the highest-energy electron in each event, if any,
    and swap it into slot 0.
    """
    pdg    = pf[:, :, pdg_idx]
    energy = pf[:, :, 0].copy()
    energy[pdg != 11] = -np.inf
    has_elec = (pdg == 11).any(axis=1)
    lead_idxs = np.argmax(energy, axis=1)
    for ev in np.where(has_elec)[0]:
        i = lead_idxs[ev]
        pf[ev, [0, i], :] = pf[ev, [i, 0], :]
    return pf


def process_particle_features(tree, feats, start, end,
                              max_part, max_nonzero,
                              status_mask=None, recompute_energy=False):
    """
    - Optionally mask the raw arrays by status_mask.
    - Pad/regularize to max_part, sort by energy, truncate to max_nonzero.
    - Optionally recompute the E component from (px,py,pz,m).
    - Always compute eta/phi and append as two new features.
    - Always swap leading e⁻ → slot 0.
    """
    # 1) pull out raw arrays, apply mask if given
    raws = [tree[feat].array()[start:end] for feat in feats]
    if status_mask is not None:
        raws = [r[status_mask] for r in raws]

    # 2) pad/fill/regularize
    parts = []
    for arr in raws:
        arr = arr.pad(max_part).fillna(0).regular()
        parts.append(arr)

    # 3) stack & sort/truncate
    pf = np.stack(parts, axis=-1)
    order = np.argsort(-pf[:, :, 0], axis=1)
    pf = np.take_along_axis(pf, order[:, :, None], axis=1)
    pf = pf[:, :max_nonzero, :]

    # 4) optional E = sqrt(px²+py²+pz²+m²)
    if recompute_energy:
        px, py, pz, m = pf[:, :, 1], pf[:, :, 2], pf[:, :, 3], pf[:, :, 8]
        pf[:, :, 0] = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    # 5) eta/phi → last two features
    eta, phi = get_eta_phi(pf[:, :, 1], pf[:, :, 2], pf[:, :, 3])
    pf = np.concatenate([pf, eta[:, :, None], phi[:, :, None]], axis=-1)

    # 6) move leading electron to slot 0
    pf = swap_leading_electron(pf, pdg_idx=9)
    return pf


def process_chunk(tmp_file, start, end, max_part, max_nonzero):
    """
    Process a chunk of events [start:end] and return reco & gen dicts.
    """
    # --- Reco inclusive kinematics ---
    reco = {
        'InclusiveKinematicsESigma':
            load_branch_arrays(tmp_file, "InclusiveKinematicsESigma",
                               kinematics_list, start, end,
                               pad=1, fill_value=0, squeeze=True),
        # add other InclusiveKinematics* as needed...
    }

    # --- Reco particle features ---
    reco['particle_features'] = process_particle_features(
        tmp_file, reco_particle_list, start, end,
        max_part, max_nonzero,
        status_mask=None, recompute_energy=False
    )

    # --- Gen inclusive kinematics ---
    gen = {
        'InclusiveKinematicsTruth':
            load_branch_arrays(tmp_file, "InclusiveKinematicsTruth",
                               kinematics_list, start, end,
                               pad=None, fill_value=None, squeeze=True)
    }

    # --- Gen particle features (truth only, status==1 + recompute E) ---
    gen_status = tmp_file['MCParticles.generatorStatus'] \
                     .array()[start:end]
    mask = (gen_status == 1)
    gen['particle_features'] = process_particle_features(
        tmp_file, gen_particle_list, start, end,
        max_part, max_nonzero,
        status_mask=mask, recompute_energy=True
    )

    return reco, gen




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
