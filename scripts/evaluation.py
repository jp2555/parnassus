import argparse
import h5py as h5
import numpy as np
import fastjet as fj
import awkward as ak

from tqdm import tqdm
from itertools import product

import concurrent.futures

import uproot

from jet_helper import Jet, get_cluster_sequence

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


# coord_var_list = ["px", "py", "pz", "pt", "eta", "phi", "mass"]
# vertex_var_list = ["vx", "vy", "vz"]

# var_dict = {
#     "events": ["event"],
#     "gens": coord_var_list + ["pdgId"] + vertex_var_list,
#     "pfcs": coord_var_list + ["pdgId"] + vertex_var_list,
# }

# prefix_dict = {
#     "gens": "GenPart_",
#     "pfcs": "PFCand_",
# }

# varlist = ["pt", "eta", "phi", "vx", "vy", "vz", "class"]
varlist = ['$x$', '$Q^2$ [GeV]$^2$','$W$', '$y$', '$\nu$', 'multiplicity']


parser = argparse.ArgumentParser(description="Preprocess full event files")
parser.add_argument("-j", "--jet", type=str, required=False, help="Input jet file")
parser.add_argument("-b", "--bkg", type=str, required=False, help="Input bkg file")
parser.add_argument("-e", "--evt", type=str, required=False, help="Input evt file")
parser.add_argument(
    "-d", "--data", type=str, required=True, help="Input test data file"
)
parser.add_argument(
    "-o", "--output", type=str, help="Output filename", default="out.root"
)
parser.add_argument("-es", "--entry_start", type=int, help="Starting event", default=0)
parser.add_argument(
    "-n", "--n_events", type=int, help="Number of events to process", default=1000
)
parser.add_argument("-dr", type=float, help="Delta R for jet clustering", default=0.5)
parser.add_argument("-no_clustering", action="store_true", help="Skip jet clustering")
parser.add_argument("-dl", action="store_true", help="Treat as delphes file")

def make_dict(data):
    data_dict = {}
    data_dict[f'pt'] = data[:,:,2]
    data_dict[f'eta'] = data[:,:,0]
    data_dict[f'phi'] = data[:,:,1]
    data_dict[f'vx'] = data[:,:,3]
    data_dict[f'vy'] = data[:,:,4]
    data_dict[f'vz'] = data[:,:,5]
    data_dict[f'class'] = np.argmax(data[:,:,7:],-1)
    mask = data_dict[f'pt'] != 0
    value_map = {0: 1, 1: 2, 2: 4, 3: 3, 4: 0}
    vectorized_map = np.vectorize(lambda x: value_map[x])
    data_dict[f'class'] = vectorized_map(data_dict[f'class'])*mask
    
    return data_dict

def load_file_h5(filename, truth_data=None,n_events=None):
    def undo_pt(x,ht):
        mask = x[:,:,2]!=0
        #x[:,:,2] = 1.0 - np.exp(x[:,:,2])
        #x[:,:,2] = np.exp(x[:,:,2])/np.sum(np.exp(x[:,:,2])*mask,1)[:,None]*ht[:,None]
        x[:,:,2] = np.exp(x[:,:,2])
        #*ht[:,None]
        return x*mask[:,:,None]

    if truth_data is None:
        truth_data = h5.File(filename,"r")['gen'][:n_events]
        truth_ht = h5.File(filename,"r")['gen_evt'][:n_events,3]
        #undo log transform for pt
        truth_data = undo_pt(truth_data,truth_ht)
        truth_data = make_dict(truth_data)
        print(truth_data['class'])
        
                
    pflow_data = h5.File(filename,"r")['reco'][:n_events]
    pflow_ht = h5.File(filename,"r")['reco_evt'][:n_events,3]
    event_number = h5.File(filename,"r")['eventNumber'][:n_events]
    pflow_data = undo_pt(pflow_data,pflow_ht)
    pflow_data = make_dict(pflow_data)


    return truth_data, pflow_data, event_number


def load_file(filename, ttype="jet", entry_start=0, n_events=None, fs=False):
    truth_data = {var: None for var in varlist}
    pflow_data = {var: None for var in varlist}
    with uproot.open(filename, num_workers=4) as file:
        tree = file[f"{ttype}_tree"]
        if n_events is None:
            n_events = tree.num_entries
        varlist_ = varlist if not fs else varlist + ["ind"]
        for var in varlist_:
            truth_data[var] = tree[f"truth_{var}"].array(
                library="np",
                entry_stop=n_events + entry_start,
                entry_start=entry_start,
            )
            pflow_data[var] = tree[f"pflow_{var}"].array(
                library="np",
                entry_stop=n_events + entry_start,
                entry_start=entry_start,
            )
        event_number = tree["eventNumber"].array(
            library="np",
            entry_stop=n_events + entry_start,
            entry_start=entry_start,
        )
    return truth_data, pflow_data, event_number


def to_ak(pt, eta, phi):
    return ak.Array(
        {
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": pt * np.sinh(eta),
            "E": pt * np.cosh(eta),
        },
        with_name="Momentum4D",
    )


def find_repeats(arr):
    return {el: np.argwhere(el == arr).flatten() for el in np.unique(arr)}


def cluster_jets(pt, eta, phi, jetdef, ptmin=20):
    particles = to_ak(pt, eta, phi)
    cs = get_cluster_sequence(
        jetdef, particles, user_indices=list(range(len(particles)))
    )
    jets = cs.inclusive_jets(ptmin=ptmin)
    jets = fj.sorted_by_pt(jets)
    jets = [Jet(j, 0.5, calc_substructure=True) for j in jets]
    jets = [j for j in jets if j.nconstituents >= 2]

    used_indices = set()

    jet_idxs = np.zeros(len(pt), dtype=int)
    for jet_idx, jet in enumerate(jets):
        particle_idx = jet.constituents_idx
        jet_idxs[particle_idx] = jet_idx
        used_indices.update(particle_idx)
    particle_idx = np.arange(len(pt))
    particle_idx = particle_idx[~np.isin(particle_idx, list(used_indices))]
    jet_idxs[particle_idx] = -1

    return jets, jet_idxs


def main():
    args = parser.parse_args()
    fevt_flag = False
    if args.evt is None:
        assert args.jet is not None and args.bkg is not None
    else:
        assert args.jet is None and args.bkg is None
        fevt_flag = True

    print(f"Loading data file: {'/'.join(args.data.split('/')[-2:])}")
    evt_truth_data, evt_pflow_data, evt_event_number = load_file_h5(args.data,n_events=args.n_events)

    if fevt_flag:
        print(f"Loading event file: {args.evt.split('/')[-1]}")
        #_, fs_data, fs_event_number = load_file(args.evt, "evt", fs=False)
        _, fs_data, fs_event_number = load_file_h5(args.evt,evt_truth_data,n_events=args.n_events)
    else:
        print(f"Loading jet file: {args.jet.split('/')[-1]}")
        _, jet_fs_data, jet_event_number = load_file(args.jet, "jet", fs=True)
        print(f"Loading bkg file: {args.bkg.split('/')[-1]}")
        _, bkg_fs_data, bkg_event_number = load_file(args.bkg, "bkg", fs=True)

    print(f"Loading data file: {'/'.join(args.data.split('/')[-2:])}")

    print("All variables loaded")

    out_data_evt = {
        f"{name}_{var}": []
        for name, var in product(["pflow", "truth", "fastsim"], varlist + ["idx"])
    }

    out_data_evt["eventNumber"] = []

    for name, var in product(
        ["pflow", "truth", "fastsim"], ["pt", "eta", "phi", "d2", "c2"]
    ):
        out_data_evt[f"{name}_jet_{var}"] = []

    jetdef = fj.JetDefinition(fj.antikt_algorithm, args.dr)

    if fevt_flag:
        goodEventNumbers = fs_event_number
    else:
        jetEventIdxs = find_repeats(jet_event_number)
        bkgEventIdxs = find_repeats(bkg_event_number)
        goodEventNumbers = np.intersect1d(jet_event_number, bkg_event_number)

    if args.n_events > len(goodEventNumbers) or args.n_events == -1:
        args.n_events = len(goodEventNumbers)
    print(
        f"Having {len(goodEventNumbers)} good events, processing {args.n_events} of them"
    )

    for i, event_number in tqdm(enumerate(goodEventNumbers), total=args.n_events):
        if i >= args.n_events:
            break
        evtEventIdx = np.argwhere(evt_event_number == event_number).flatten()
        truth_data = {var: evt_truth_data[var][evtEventIdx][0] for var in varlist}
        
        truth_evt_mask = (
            (truth_data["pt"] > 0)
            & (truth_data["pt"] < 1e4)
            & (np.abs(truth_data["eta"]) < 3)
        )
        for key, val in truth_data.items():
            truth_data[key] = val[truth_evt_mask]


        pflow_data = {var: evt_pflow_data[var][evtEventIdx][0] for var in varlist}
        pflow_evt_mask = (
            (pflow_data["pt"] > 1)
            & (pflow_data["pt"] < 1e4)
            & (np.abs(pflow_data["eta"]) < 3)
        )
        for key, val in pflow_data.items():
            pflow_data[key] = val[pflow_evt_mask]


        if fevt_flag:
            fsEventIdx = np.argwhere(fs_event_number == event_number).flatten()
            fs_evt_data = {
                var: np.concatenate(fs_data[var][fsEventIdx]) for var in varlist
            }


            fs_evt_mask = (
                (fs_evt_data["pt"] > 1)
                & (fs_evt_data["pt"] < 1e4)
                & (np.abs(fs_evt_data["eta"]) < 3)
            )
            for key, val in fs_evt_data.items():
                fs_evt_data[key] = val[fs_evt_mask]
        else:
            jetEventIdx = jetEventIdxs[event_number]
            bkgEventIdx = bkgEventIdxs[event_number]

            fs_jet_ind = np.concatenate(jet_fs_data["ind"][jetEventIdx]).astype(bool)
            fs_bkg_ind = np.concatenate(bkg_fs_data["ind"][bkgEventIdx]).astype(bool)
            fs_evt_data = {
                var: np.concatenate(
                    [
                        np.concatenate(jet_fs_data[var][jetEventIdx])[fs_jet_ind],
                        np.concatenate(bkg_fs_data[var][bkgEventIdx])[fs_bkg_ind],
                    ],
                    axis=0,
                )
                for var in varlist
            }
            fs_evt_data["pt"] = fs_evt_data["pt"] / 1000
            fs_evt_mask = (
                (fs_evt_data["pt"] > 1)
                #& (fs_evt_data["pt"] < 1e4)
                & (np.abs(fs_evt_data["eta"]) < 3)
            )
            for key, val in fs_evt_data.items():
                fs_evt_data[key] = val[fs_evt_mask]


        pf_jets, pf_jet_indices = cluster_jets(
            pflow_data["pt"], pflow_data["eta"], pflow_data["phi"], jetdef
        )
        tr_jets, tr_jet_indices = cluster_jets(
            truth_data["pt"], truth_data["eta"], truth_data["phi"], jetdef
        )
        fs_jets, fs_jet_indices = cluster_jets(
            fs_evt_data["pt"], fs_evt_data["eta"], fs_evt_data["phi"], jetdef
        )

        for key in varlist:
            out_data_evt[f"truth_{key}"].append(truth_data[key])
            out_data_evt[f"pflow_{key}"].append(pflow_data[key])
            out_data_evt[f"fastsim_{key}"].append(fs_evt_data[key])
        out_data_evt["truth_idx"].append(tr_jet_indices)
        out_data_evt["pflow_idx"].append(pf_jet_indices)
        out_data_evt["fastsim_idx"].append(fs_jet_indices)

        for name, jet_collection in zip(
            ["pflow", "truth", "fastsim"], [pf_jets, tr_jets, fs_jets]
        ):
            out_data_evt[f"{name}_jet_pt"].append(
                np.array([jet.pt() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_eta"].append(
                np.array([jet.eta() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_phi"].append(
                np.array([jet.phi() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_d2"].append(
                np.array([jet.substructure["d2"] for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_c2"].append(
                np.array([jet.substructure["c2"] for jet in jet_collection])
            )

        out_data_evt["eventNumber"].append(event_number)

    with uproot.recreate(
        f"{args.output.replace('.root', '')}_{args.n_events//1000}k.root"
    ) as f:
        f["evt_tree"] = {
            "truth": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("truth") and "jet" not in key
                }
            ),
            "pflow": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("pflow") and "jet" not in key
                }
            ),
            "fastsim": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("fastsim") and "jet" not in key
                }
            ),
            "eventNumber": out_data_evt["eventNumber"],
            "truth_jets": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("truth_jet") and "idx" not in key
                }
            ),
            "pflow_jets": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("pflow_jet") and "idx" not in key
                }
            ),
            "fastsim_jets": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val)
                    for key, val in out_data_evt.items()
                    if key.startswith("fastsim_jet") and "idx" not in key
                }
            ),
        }


if __name__ == "__main__":
    main()






