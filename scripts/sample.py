import os
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
import argparse
import pickle
from PET import PET
import utils
import plot_utils
import matplotlib.pyplot as plt
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#keeping track of the number of variables plotted

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process elec data.")
    parser.add_argument("--dataset", type=str, default="parnassus", help="Dataset to use")
    parser.add_argument("--folder", default="/pscratch/sd/v/vmikuni/parnassus/", help="Folder containing input files")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--nevts", type=int, default=-1, help="Number of events to load")
    parser.add_argument("--sample", action='store_true', help="Sample from trained model")
    parser.add_argument("--plot_folder", default="../plots", help="Folder to save the outputs")
    parser.add_argument("--val_file", default="val_ggF", help="Folder to save the outputs")
    parser.add_argument("--name", default="parnassus", help="File to save the outputs")
    
    parser.add_argument("--K", type=int, default=3, help="K neighbors")
    parser.add_argument("--num_local", type=int, default=1, help="number of local layers for knn")    
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--projection", type=int, default=128, help="base projection size")

    return parser.parse_args()

def get_data_info(flags):
    test = utils.DataLoader(os.path.join(flags.folder,f'{flags.val_file}.h5'), flags.batch, hvd.rank(), hvd.size())
    return test

def get_model_name(flags):
    model_name = f'parnassus_qcd_{flags.K}_{flags.num_local}_{flags.num_layers}_{flags.projection}.weights.h5'
    return model_name


def load_data_and_model(flags):
    
    test = get_data_info(flags)
    model = PET(num_feat=test.num_feat,
                num_evt=test.num_evt,
                num_part=test.num_part,
                projection_dim = flags.projection,
                K = flags.K,
                num_layers = flags.num_layers,
                num_local = flags.num_local,
                )
    
    model_name = os.path.join(flags.folder, 'checkpoints', get_model_name(flags))
    print(f"loading model {model_name}")
    model.load_weights(model_name)
    return test, model


def sample_data(test, model, flags, sample_name):
    """ Sample data using the model and save to file. """
    
    nsplit = 200
    gen_part,gen_mask,gen_evt = test.get_preprocess_cond(flags.nevts)
    p, j = model.generate(gen_part,gen_mask,gen_evt,
                          nsplit=nsplit,use_tqdm=hvd.rank()==0)
    p = test.revert_preprocess(p, p[:, :, 2] != 0)
    j = test.revert_preprocess_evt(j)

    del gen_part, gen_mask, gen_evt
    gc.collect()
    reco = hvd.allgather(tf.constant(p)).numpy()
    reco_evt = hvd.allgather(tf.constant(j)).numpy()

    if hvd.rank() == 0:
        with h5.File(sample_name, "w") as h5f:
            h5f.create_dataset("reco", data=reco)
            h5f.create_dataset("reco_evt", data=reco_evt)
                        
def get_generated_data(sample_name,nevts=-1):

    with h5.File(sample_name,"r") as h5f:
        if nevts>0:
            nevts = None
        reco_evt = h5f['reco_evt'][:nevts]
        reco_particles = h5f['reco'][:nevts]

        
    def undo_pt(x):
        x[:,:,2] = 1.0 - np.exp(reco_particles[:,:,2])
        x[:,:,2] = x[:,:,2]/np.sum(x[:,:,2],1,keepdims=True)
        x[:,:,2] = x[:,:,2]*reco_evt[:,-1,None]
        return x

    mask = reco_particles[:,:,2]!=0
    #undo log transform for pt
    reco_particles = undo_pt(reco_particles)
    reco_particles = reco_particles*mask[:,:,None]
    return reco_evt, reco_particles


def get_from_dataloader(test,nevts=-1):
    #Load eval samples for metric calculation    
    reco_data,_,reco_mask,_,reco_evt, _ = test.data_from_file(test.files[0],nevts=nevts,preprocess=False)

    reco_data[:,:,2] = 1.0 -np.exp(reco_data[:,:,2])
    reco_data[:,:,2] = reco_data[:,:,2]/np.sum(reco_data[:,:,2],1,keepdims=True)
    reco_data[:,:,2] = reco_data[:,:,2]*reco_evt[:,-1,None]*reco_mask
    #only keep the first 3 features
    # if nevts<0:
    #     nevts = reco_evt.shape[0]
        
    # reco_data = reco_data[:nevts]
    # reco_evt = reco_evt[:nevts]
    return reco_evt, reco_data


def plot(evt1,evt2,var_names,title,plot_folder):
    for ivar in range(len(var_names)):                
        feed_dict = {
            'cms_truth':evt1[:,ivar],
            'cms_gen':  evt2[:,ivar]
        }
            
        
        fig,gs,binning = plot_utils.HistRoutine(feed_dict,xlabel=var_names[ivar],
                                                plot_ratio=True,
                                                reference_name='cms_truth',
                                                logy = ivar >= 2 and 'Particle' in title,
                                                binning = np.linspace(0,35,30) if ivar == 2 and 'Particle' in title else None,
                                                ylabel= 'Normalized entries')

        fig.savefig('{}/Parnassus_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')
        plt.close()


def plot_results(evts, evts_gen, particles, particles_gen, flags):
    """ Plot the results using the utility functions. """

    var_names = ['Multiplicity','$MET_x$ [GeV]','$MET_y$ [GeV]','HT [GeV]']

    plot(evts, evts_gen, title='Event_{}'.format(flags.val_file),
         var_names=var_names, plot_folder=flags.plot_folder)
    
    #Mask zero-padded particles
    particles_gen=particles_gen.reshape((-1,particles_gen.shape[-1]))
    particles_gen=particles_gen[particles_gen[:,2]!=0.]
    particles=particles.reshape((-1,particles.shape[-1]))
    particles=particles[particles[:,2]!=0.]
    
    #Inclusive plots with all particles
    part_var_names = ['$\eta$', '$\phi$', '$p_{Trel}$',
                      'vx','vy','vz','charge',
                      'is electron','is muon', 'is photon',
                      'is neutral hadron','is charged hadron']
    
    plot(particles, particles_gen,title='Particle_{}'.format(flags.val_file),
         var_names=part_var_names,plot_folder=flags.plot_folder)
    
def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank()==0:logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()
    sample_name = os.path.join(flags.folder, f'{flags.name}.h5')
    
    if flags.sample:
        if hvd.rank()==0:logging.info("Sampling the data.")
        test, model = load_data_and_model(flags)
        sample_data(test, model, flags, sample_name)
    else:
        if hvd.rank()==0:logging.info("Loading saved samples.")
        # Load and process data, generate plots, etc.        
        test = get_data_info(flags)
        evts, particles = get_from_dataloader(test,flags.nevts)
        evts_gen, particles_gen = get_generated_data(sample_name,flags.nevts)
        print(particles_gen.shape,particles.shape)
        # Plot results
        plot_results(evts, evts_gen, particles, particles_gen, flags)

if __name__ == '__main__':
    main()


