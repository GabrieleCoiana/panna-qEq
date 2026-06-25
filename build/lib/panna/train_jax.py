import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import time
import argparse
import logging
import glob
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import tensorflow as tf
import copy
# Preventing TF from using the GPU
# as it should only handle data preprocessing
tf.config.set_visible_devices([], 'GPU')
from functools import partial

logger = logging.getLogger('panna')
from panna.lib.log import emit_splash_screen

from panna.jax.parser_jax import trainjax_parameter_parser
from panna.jax.data_jax import create_dataset
from panna.jax.model_jax import make_model
from panna.jax.val_jax import validate
from panna.jax.losses_jax import compute_loss, compute_diffsums
from panna.jax.metrics_jax import metrics

def train_step(parameters, modapp, tx, make_loss, model_state, opt_state, batch):
    fixed_vars = {}
    # Removing variables that should not be optimized
    # so we don't even compute the gradients
    for v in parameters['fixed_variables']:
        fixed_vars[v] = model_state.pop(v)
        if len(opt_state[0])>1:
            if v in opt_state[0][1]:
                opt_state[0][1].pop(v)
            if len(opt_state[0])>2:
                if v in opt_state[0][2]:
                    opt_state[0][2].pop(v)
    (loss, fom), grads = make_loss(model_state, fixed_vars, batch)

    # Adamw also wants the state passed...
    if parameters['l1_coef'] > 0.0:
        updates, opt_state = tx.update(grads, opt_state, model_state)
    else:
        updates, opt_state = tx.update(grads, opt_state)
    model_state = optax.apply_updates(model_state, updates)

    # Reintroducing fixed variables
    for v in parameters['fixed_variables']:
        model_state[v] = fixed_vars[v]
    # Clipping parameters that need it
    if 'clipping_vars' in parameters.keys() and len(parameters['clipping_vars'])>0:
        for cd in parameters['clipping_vars']:
            # Clipping type 1: [min, max]
            if cd[0]==1:
                model_state[cd[1]][cd[2]] = jnp.clip(
                                                model_state[cd[1]][cd[2]],
                                                cd[3],cd[4])
            # Clipping type 2: centers need to refer to their sigmas
            # we clip from sigma to Rc-sigma
            elif cd[0]==2:
                maxvals = cd[4]-model_state[cd[1]][cd[3]]
                model_state[cd[1]][cd[2]] = jnp.clip(
                                                model_state[cd[1]][cd[2]],
                                                model_state[cd[1]][cd[3]],maxvals)
            # Clipping type 3: we compare cent+sigma to a cutoff matrix per species
            # and set to zero elements where cent+sig > Rc
            elif cd[0]==3:
                maxvals = jnp.expand_dims(model_state[cd[1]][cd[3]]+model_state[cd[1]][cd[4]],1)
                model_state[cd[1]][cd[2]] = jnp.where(maxvals>cd[5],0.0,model_state[cd[1]][cd[2]])
    return model_state, opt_state, loss, fom

def init_train(parameters, model, grad_loss, batch):
    e = 0
    s = 0
    opt_state = None
    # Parsing possible restart 
    if parameters['load_weights']:
        # If the model is specified
        if parameters['weights_file']:
            if os.path.exists(parameters['weights_file']):
                weif = parameters['weights_file']
                print('Weights file found.')
                optf = None
                print('Please note that the optimizer schedule will start from scratch.')
            else:
                print('Could not find model file: '+
                          parameters['weights_file']+'. Aborting.')
                exit()
        else:
            # Looking for last model
            print('Looking for the last model in {}/models'.format(
                                     parameters['train_dir']))
            files = glob.glob(parameters['train_dir']+'/models/*pkl')
            if len(files)>0:
                namest = [(f,int(f.split('step_')[-1].split('.')[0])) \
                          for f in files \
                          if f.split('step_')[-1].split('.')[0].isnumeric()]
                weif = namest[np.argmax([n[1] for n in namest])][0]
                optf = parameters['train_dir']+'/models/opt_state.pkl'
                if not os.path.exists(optf):
                    print('Optimizer state not found. Aborting.')
                    exit()
            else:
                print('Could not find any model file. '+
                               'Restarting from scratch')
                weif = None
                optf = None
        # Actually loading the weights
        if weif:
            print('Loading weights from '+weif)
            with open(weif, 'rb') as f:
                model_state = pickle.load(f)
            if optf:
                with open(optf, 'rb') as f:
                    opt_state = pickle.load(f)
            # Establish epoch/step for consistency
            trial_e = weif.split('epoch_')[-1].split('_')[0]
            if trial_e.isnumeric():
                e = int(trial_e)
            trial_s = weif.split('step_')[-1].split('.')[0]
            if trial_s.isnumeric():
                s = int(trial_s)
            print('Restarting from epoch {}, step {}'.format(e,s))

    if (not parameters['load_weights']) or weif==None:
        # Initializing optimizer
        if parameters['rand_seed']:
            rngkey = jax.random.PRNGKey(np.random.randint(100000))
        else:
            rngkey = jax.random.PRNGKey(42)
        print('Initializing random weights with key: ',rngkey)
        model_state = model.init(rngkey, batch)
    else:
        _ = model.init(0, batch)

    # Parsing loss schedule
    sch_list = []
    sch_bounds = []
    for sch_str in parameters['learning_rate']:
        sch_pars = sch_str.split(':')
        # Options: lr (single constant lr)
        if len(sch_pars)==1:
            sch_list.append(optax.constant_schedule(float(sch_pars[0])))
        # Option: 'const:lr:(bound)'
        elif sch_pars[0]=='const':
            sch_list.append(optax.constant_schedule(float(sch_pars[1])))
            if len(sch_pars)>2:
                sch_bounds.append(int(sch_pars[2]))
        # Option: 'exp:lr:decay:steps'
        elif sch_pars[0]=='exp':
            sch_list.append(optax.exponential_decay(init_value=float(sch_pars[1]),
                transition_steps=int(sch_pars[3]),
                decay_rate=float(sch_pars[2])))
            sch_bounds.append(int(sch_pars[3]))
    if len(sch_list)==1:
        schedule = sch_list[0]
    else:
        schedule = optax.join_schedules(sch_list, boundaries=np.cumsum(sch_bounds)[:-1])
    
    # Creating optimizer
    if parameters['l1_coef'] > 0.0:
        tx = optax.adamw(schedule, weight_decay=parameters['l1_coef'])
    else:
        tx = optax.adam(schedule)
    # Initializing optimizer if we did not load it
    if not opt_state:
        fixed_vars = {}
        # Removing variables that should not be optimized
        for v in parameters['fixed_variables']:
            fixed_vars[v] = model_state.pop(v)
        opt_state = tx.init(model_state)
        for v in parameters['fixed_variables']:
            model_state[v] = fixed_vars[v]

    # Creating optimized training function
    train_func = jax.jit(partial(train_step, parameters, model.apply, tx, grad_loss))
    
    return train_func, model_state, tx, opt_state, e, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default='',
                        help="config file",
                        required=True)
    flags, _unparsed = parser.parse_known_args()
    emit_splash_screen(logger)
    # Parsing config file
    parameters = trainjax_parameter_parser(flags.config)
    train_f = parameters['forces']

    # Creating dataset
    data, Vdata, pre = create_dataset(parameters)

    # Creating model
    raw_model = make_model(parameters, pre=pre)
    model = hk.without_apply_rng(hk.transform(raw_model))
    modapp = jax.jit(model.apply)
    grad_loss = jax.value_and_grad(partial(
                        compute_loss, parameters, modapp), has_aux=True)

    # Jump to val code if we only want validation
    if parameters['task']=='val' or parameters['task']=='descr':
        validate(parameters, Vdata, modapp)
        exit()

    # Init steps, outputs, metrics
    s = 0
    e = 0
    adt = None
    mix = parameters['mix_factor']
    omix = 1-mix
    if not os.path.exists(parameters['train_dir']):
        os.mkdir(parameters['train_dir'])
    if not os.path.exists(parameters['train_dir']+'/models'):
        os.mkdir(parameters['train_dir']+'/models')
    met = metrics(parameters)
    if not os.path.exists(parameters['train_dir']+'/metrics.dat'):
        with open(parameters['train_dir']+'/metrics.dat', 'w') as f:
            f.write(met.header())
    ff = open(parameters['train_dir']+'/metrics.dat', 'a')
    tt = time.time()
    pt = tt-2
    et = time.time()

    # Main loop
    for d in data:
        # Get a training batch
        batch = {k:d[k].numpy() for k in d.keys() if k!='name'}
        # Init model at step 0 with the first batch
        if s==0:
            train_func, model_state, tx, opt_state, e, s = \
                init_train(parameters, model, grad_loss, batch)
        # Training step
        try:
            model_state, opt_state, loss, fom = \
                train_func(model_state, opt_state, batch)
        except RuntimeError as re:
            print('Runtime error: '+str(re)[:300]+'... Probably out of memory.')
            print('Skipping this batch. If the error persists reduce parameters.')
            continue

        ttt = time.time()
        dt = ttt-tt
        # Updating command line figures of merit
        if adt==None:
            met.update_train(fom, first=True)
            aloss = loss
            adt = dt
        else:
            met.update_train(fom)
            aloss = omix*aloss + mix*loss
            adt = omix*adt + mix*dt
        # Print to command line at most once per second
        if ttt-pt>1:
            pt = ttt
            ETA = adt*(parameters['epoch_size']-s%parameters['epoch_size'])
            ETAs = time.strftime("%H:%M:%S", time.gmtime(int(ETA)))
            print('Epoch: ', e+1, 'step: ', s, 'dt: {:.3f}'.format(adt*1000), 'ms ETA: ', ETAs,' loss: ', aloss, end=' ')
            print(met.train_string(), end='\r', flush=True)
        tt = ttt
        s += 1
        
        # End-of-epoch computations
        if (s%parameters['epoch_size']==0):
            # Updating command line with epoch time
            etime = int(ttt - et)
            etime = time.strftime("%H:%M:%S", time.gmtime(etime))
            et = ttt
            print('Epoch: ', e+1, 'step: ', s, 'dt: {:.3f}'.format(adt*1000), 'ms Epoch time: ', etime,' loss: ', aloss, end=' ')
            print(met.train_string(), end='\n', flush=True)
            e +=1

            # Saving the model
            if e%parameters['save_epoch_freq']==0:
                with open(parameters['train_dir']+
                     '/models/epoch_{}_step_{}.pkl'.format(e,s), 'wb') as f:
                    pickle.dump(model_state, f)
                # Save optimizer state
                with open(parameters['train_dir']+
                     '/models/opt_state.pkl', 'wb') as f:
                    pickle.dump(opt_state, f)

            # Validation on the whole val dataset
            if Vdata:
                # Running over batches and accumulating sums for proper average
                for dat in Vdata:
                    vbatch = {k:dat[k].numpy() for k in dat.keys() if k!='name'}
                    vfom = compute_diffsums(model_state, modapp, parameters, vbatch)
                    met.update_val(vfom)
                vstr, vout = met.finalize_val()
                print('         '+vstr, flush=True)

            # Saving foms to file
            outq = [e,s]
            outq.extend(met.train_list())
            if Vdata:
                outq.extend(vout)
            ff.write('\t'.join([str(n) for n in outq])+'\n')
            ff.flush()

            # If we finished the last epoch, get out
            if e==parameters['epochs']:
                break
            
            # Skipping validation time in the timer..
            tt =  time.time()

    ff.close()
    print("Training done!")

if __name__ == '__main__':
    main()
