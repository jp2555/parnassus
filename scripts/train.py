import os
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse
import logging
import pickle
import wandb

from wandb.integration.keras import WandbMetricsLogger

# Custom local imports
import utils
from PET import PET,PETCorrector

# Keras imports
from tensorflow.keras.optimizers import schedules, Lion
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Initialize Horovod
hvd.init()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_name(flags):
    if flags.corrector:
        model_name = f'parnassus_qcd_{flags.K}_{flags.num_local}_{flags.num_layers}_{flags.projection}_corrector.weights.h5'
    else:
        model_name = f'parnassus_qcd_{flags.K}_{flags.num_local}_{flags.num_layers}_{flags.projection}.weights.h5'
    return model_name

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the PET model.")
    parser.add_argument("--dataset", type=str, default="parnassus", help="Dataset to use")
    parser.add_argument("--folder", type=str, default="/mscratch/sd/v/vmikuni/parnassus/", help="Folder containing input files")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--epoch", type=int, default=300, help="Max epoch")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=1.0, help="Factor to adjust learning rate")
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune a model')
    parser.add_argument("--corrector", action='store_true', default=False, help='Learn a linear correction to generated events')
    parser.add_argument("--load", action='store_true', help="Continue the training")
    
    parser.add_argument("--K", type=int, default=3, help="K neighbors")
    parser.add_argument("--num_local", type=int, default=1, help="number of local layers for knn")    
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--projection", type=int, default=128, help="base projection size")
    
    
    return parser.parse_args()

def configure_optimizers(flags, train_loader,lr_factor = 1.0):
    scale_lr = flags.lr * 2
    lr_schedule = schedules.CosineDecay(
        initial_learning_rate=flags.lr/lr_factor,
        warmup_target=scale_lr/lr_factor,
        warmup_steps=3*train_loader.nevts//flags.batch//hvd.size(),
        decay_steps=flags.epoch*train_loader.nevts//flags.batch//hvd.size(),
    )
    optimizer = Lion(
        learning_rate=lr_schedule,
        #clipnorm=1.0,
        beta_1=0.95,
        beta_2=0.99,
        weight_decay = 1e-2
    )
    return hvd.DistributedOptimizer(optimizer)

def main():
    utils.setup_gpus()
    flags = parse_arguments()

    if flags.corrector:
        train_loader = utils.DataLoader(os.path.join(flags.folder),
                                        reference = 'ggF.h5', correction = 'parnassus_ggF.h5',
                                        batch_size = flags.batch,
                                        rank = hvd.rank(), size = hvd.size(),
                                        corrector = True)
        val_loader = utils.DataLoader(os.path.join(flags.folder),
                                      reference = 'ggF.h5', correction = 'parnassus_ggF.h5',
                                      batch_size = flags.batch,
                                      rank = hvd.rank(), size = hvd.size(),
                                      corrector = True)

    else:
        train_loader = utils.DataLoader(os.path.join(flags.folder,'h5'),
                                        names = ['top','qcd_400','qcd_600'],
                                        batch_size = flags.batch,
                                        rank = hvd.rank(), size = hvd.size())
        val_loader = utils.DataLoader(os.path.join(flags.folder,'h5'),
                                      names = ['ggF'],
                                      batch_size = flags.batch,
                                      rank = hvd.rank(), size = hvd.size())

        
    if flags.fine_tune:
        model_name = 'parnassus_v2_qcd_pretrain'
        model_name = os.path.join(flags.folder,'checkpoints',model_name)
    else:
        model_name = None

    if hvd.rank() ==0:
        # mode = 'disabled'
        wandb.login()
        mode = None
    else:
        mode = 'disabled'
        
    run = wandb.init(
        # Set the project where this run will be logged
        project="Parnassus_v3",
        name=get_model_name(flags).replace(".weights.h5",""),
        mode = mode,
        # Track hyperparameters and run metadata            
        config={
            "learning_rate": flags.lr,
            "epochs": flags.epoch,
            "batch size": flags.batch,
        },
    )


    if flags.corrector:
        model = PETCorrector(num_feat=train_loader.num_feat,
                             projection_dim = flags.projection,
                             K = flags.K,
                             num_layers = flags.num_layers,
                             num_local = flags.num_local,
                             #Corrector learns a linear combination of the outputs ax+b
                             )

    else:
        model = PET(num_feat=train_loader.num_feat,
                    num_evt=train_loader.num_evt,
                    num_part=train_loader.num_part,
                    projection_dim = flags.projection,
                    K = flags.K,
                    num_layers = flags.num_layers,
                    num_local = flags.num_local,
                    )

    if flags.load:
        if hvd.rank()==0:
            model_name = get_model_name(flags)
            model_path = os.path.join(flags.folder, 'checkpoints', model_name)
            logger.info(f"Loading model weights from {model_path}")
            model.load_weights(model_path,by_name=True,skip_mismatch=True)

    
    optimizer_body = configure_optimizers(flags, train_loader, lr_factor=flags.lr_factor if flags.fine_tune else 1)
    optimizer_head = configure_optimizers(flags, train_loader)
    model.compile(optimizer_body, optimizer_head)

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        EarlyStopping(patience=200, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=300, min_lr=1e-6)
    ]

    if hvd.rank() == 0:
        checkpoint_name = get_model_name(flags)
        checkpoint_path = os.path.join(flags.folder, 'checkpoints', checkpoint_name)
        checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                              save_best_only=True, mode='auto',
                                              save_weights_only=True,
                                              period=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(WandbMetricsLogger())
        

    hist = model.fit(train_loader.make_tfdata(),
                     epochs=flags.epoch,
                     validation_data=val_loader.make_tfdata(),
                     batch_size=flags.batch,
                     callbacks=callbacks,
                     steps_per_epoch=train_loader.steps_per_epoch,
                     validation_steps=val_loader.steps_per_epoch,
                     verbose=hvd.rank() == 0)


if __name__ == "__main__":
    main()

