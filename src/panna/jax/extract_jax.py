import os
import argparse
import logging
import pickle
import numpy as np
logger = logging.getLogger('panna')
from panna.lib.log import emit_splash_screen

from parser_jax import trainjax_parameter_parser

act2num = {'exp': '0'}

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
    if parameters['descriptor_type']=='PNE':
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
            f.write('model_file=model.bin\n')
            # TODO pars['mlp_sp_weights']
        if parameters['descriptor_type']=='PNE':
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

    # Saving weights
    # Serializing descriptor weights
    modw = np.array([], dtype=np.float32)
    if parameters['descriptor_type']=='PNE':
        dekey = [k for k in model_state.keys() if 'h_kpre_list' in k]
        if len(dekey)==1:
            dedict = model_state[dekey[0]]
        else:
            raise ValueError('Could not find descriptor weights in model file. Aborting.')
        # Because we have 2 notations
        if parameters['wname']=='new':
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
            raise ValueError('Old descr storage not supported yet!')
    else:
        raise ValueError('Descriptor'+parameters['descriptor_type']+\
                         'is not supported yet!')

    # Serializing model weights
    if parameters['model_type']=='MLP':
        lays = ['_'+str(i) for i in range(len(parameters['mlp_arch']))]
        lays[0] = ''
        for nl, l in enumerate(lays):
            try:
                key = 'panna_mlp_list/~/atom__linear'+l
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


if __name__ == '__main__':
    main()