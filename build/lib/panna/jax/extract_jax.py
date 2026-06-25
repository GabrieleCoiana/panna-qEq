import os
import argparse
import logging
import pickle
import numpy as np
import glob
try:
    import torch
except:
    pass

logger = logging.getLogger('panna')
from panna.lib.log import emit_splash_screen

from parser_jax import trainjax_parameter_parser
from model_torch import get_scripted_model_torch

act2num = {'exp': '0'}
fmt2num = {'bin': '0', 'torch': '1'}

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

    if parameters['extract_dir']==None:
        parameters['extract_dir'] = '.'
    print("Saving weights for lammps to ",parameters['extract_dir'])
    if not os.path.exists(parameters['extract_dir']):
        os.mkdir(parameters['extract_dir'])
    # Get size of descriptor:
    if parameters['descriptor_type']=='LATTE':
        dsize = np.sum([int(c.split(',')[0]) for c in \
                   parameters['descriptor_shape'].split(':')])
    # Save parameters
    with open(parameters['extract_dir']+'/panna.in', 'w') as f:
        f.write('species='+','.join([str(len(parameters['species']))]+\
                           parameters['species'])+'\n')
        f.write('model='+parameters['model_type']+'\n')
        if parameters['model_type']=='MLP':
            # First the #layers, then layers, including descr
            f.write('architecture='+','.join(\
                [str(len(parameters['mlp_arch'])+1),str(dsize)]+\
                [str(l) for l in parameters['mlp_arch']])+'\n')
            try:
                f.write('activation='+act2num[parameters['mlp_act']]+'\n')
            except:
                raise ValueError('Unsupported activation.')
            if parameters['extract_format']=='bin':
                f.write('model_file=model.bin\n')
                # Continuous species not implemented in binary
                f.write('cont_sp=0\n')
            elif parameters['extract_format']=='torch':
                f.write('model_file=model.pt\n')
                if parameters['extf']=='std' or parameters['extf']=='cont_sp':
                    f.write('cont_sp=1\n')
                else:
                    f.write('cont_sp=0\n')
            # TODO pars['mlp_sp_weights']
        if parameters['descriptor_type']=='LATTE':
            f.write('descriptor=LATTE\n')
            f.write('cutoff={}\n'.format(parameters['cutoff']))
            # Notation is:
            # num_blocks:block1:....:block_n
            # Each block:
            # num_elements,max_dims,num_bodies
            #   Then only if it's not '-':
            #   [num_inds_1(..inds_1..),...,num_inds_b(..inds_b)]
            # f.write('descriptor_shape='+\
            #     str(len(parameters['descriptor_shape'].split(':')))+':'+\
            #         ':'.join([','.join([b.split(',')[0],str(len(b.split(','))-1)]+b.split(',')[1:]) \
            #              for b in parameters['descriptor_shape'].split(':')])+'\n')
            dstring = str(len(parameters['descriptor_shape'].split(':')))
            for ds in parameters['descriptor_shape'].split(':'):
                parts = ds.split(',')
                dstring += ':'+parts[0]+','
                inds = list(set([c for p in parts[1:] for c in p if c!='-']))
                inds.sort()
                max_dims = len(inds)
                dstring += str(max_dims)+','
                n_bodies = len(parts)-1
                dstring += str(n_bodies)
                if max_dims>0:
                    dstring += '['
                    for p in parts[1:]:
                        dims = [c for c in p]
                        dstring += str(len(dims))+'('
                        dstring += ','.join([str(inds.index(c)) for c in dims])
                        dstring += ',),'
                    dstring += ']'
            f.write('descriptor_shape='+dstring+'\n')
            if not parameters['learnsig']:
                f.write('sigma={}\n'.format(parameters['sigma']))
            f.write('format='+fmt2num[parameters['extract_format']]+'\n')
            # TODO spec_weights, sp_emb

    # Loading weights
    if parameters['weights_file']:
        if os.path.exists(parameters['weights_file']):
            weif = parameters['weights_file']
            print('Weights file found.')
        else:
            raise FileNotFoundError('Could not find model file: '+
                      parameters['weights_file']+'. Aborting.')
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
        else:
            raise FileNotFoundError('Could not find any model file. Aborting.')
    with open(weif, 'rb') as f:
        model_state = pickle.load(f)

    # Handling multiple targets for last layer
    if parameters['extract_target']>-1:
        targ = parameters['extract_target']-1
        if parameters['model_type']=='MLP':
            key = 'panna_mlp/~/atom__linear_'+str(len(parameters['mlp_arch'])-1)
            if parameters['mlp_sp_weights']:
                sp_w = parameters['mlp_sp_weights'].split(':')[-1].strip()
            else:
                sp_w = 's'
            if sp_w=='s':
                model_state[key]['w'] = model_state[key]['w'][:,targ:targ+1]
                model_state[key]['b'] = model_state[key]['b'][:,targ:targ+1]
            elif sp_w=='c':
                model_state[key]['w'] = model_state[key]['w'][targ:targ+1]
                model_state[key]['b'] = model_state[key]['b'][targ:targ+1]
            elif sp_w=='e':
                model_state[key]['w'] = model_state[key]['w'][targ:targ+1]
                model_state[key]['b'] = model_state[key]['b'][:,targ:targ+1]
        parameters['offsets'] = parameters['offsets'][targ]


    # Saving weights
    # Binary format
    if parameters['extract_format']=='bin':
        # Serializing descriptor weights
        modw = np.array([], dtype=np.float32)
        if parameters['descriptor_type']=='LATTE':
            dekey = [k for k in model_state.keys() if 'h_kpre' in k]
            if len(dekey)==1:
                dedict = model_state[dekey[0]]
            else:
                raise ValueError('Could not find descriptor weights in model file. Aborting.')
            for elem in parameters['descriptor_shape'].split(':'):
                els = ','.join(elem.split(',')[1:])
                try:
                    key = 'DV_spw_'+els
                    modw = np.append(modw,dedict[key].flatten())
                    key = 'DV_cent_'+els
                    modw = np.append(modw,dedict[key].flatten())
                    if parameters['learnsig']:                        
                        key = 'DV_sig_'+els
                        modw = np.append(modw,dedict[key].flatten())
                except:
                    raise ValueError('Could not find key: '+key)
        else:
            raise ValueError('Descriptor '+parameters['descriptor_type']+\
                             ' is not supported yet!')

        # Serializing model weights
        if parameters['model_type']=='MLP':
            lays = ['_'+str(i) for i in range(len(parameters['mlp_arch']))]
            lays[0] = ''
            for nl, l in enumerate(lays):
                try:
                    key = 'panna_mlp/~/atom__linear'+l
                    for part in ['w','b']:
                        mat = model_state[key][part].flatten()
                        # Adding offsets for the last term
                        if nl==len(lays)-1 and part=='b':
                            mat += np.asarray(parameters['offsets'])
                        modw = np.append(modw,mat)
                except:
                    raise ValueError('Could not find key: '+key+\
                                     ', element '+part)

        # Dumping all weights
        with open(parameters['extract_dir']+'/model.bin', 'wb') as f:
            modw.tofile(f)
        print('All weights saved.')

    # Torch model format
    elif parameters['extract_format']=='torch':
        # General idea...
        scripted_model = get_scripted_model_torch(parameters, model_state)
        torch.jit.save(scripted_model, parameters['extract_dir']+'/model.pt')
        print('Torch model saved.')


if __name__ == '__main__':
    main()