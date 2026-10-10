from panna.neuralnet.inputs_iterator import input_pipeline, input_pipeline_lr
from panna.gvector.gvect_latte import GvectLATTE
from panna.lib import ExampleJsonWrapper
import os
import numpy as np
import psutil
import jax
from functools import partial


# Dummy class since we don't need the complete ParseFn
class dummyPF():
    def __init__(self,
                 forces: bool = False):
        self._forces = forces

    def __call__(self):
        return None


def create_dataset_new(parameters):
    forces = parameters['forces_cost'] > 0.0
    long_range = 'LR' in parameters['model_type'] # parameters['model_type'] in ['MLP_LR','MLP_LR_fq','MLP_LR_mod', 'MLP_LR_modsr', 'MLP_LR_fq_modsr']
    long_range_pol = 'pol' in parameters['model_type'] # parameters['model_type'] in ['MLP_LR_pol' ,'MLP_LR_fq_pol', 'MLP_LR_fq_pol_modsr']
    charges = parameters['charges_cost'] > 0.0
    dipoles = parameters['dipoles_cost'] > 0.0
    stress = parameters['stress_cost'] > 0.0
    

    # Dummy parseFn
    PF = dummyPF(forces)
    
    # This works with LATTE for now
    if parameters['descriptor_type'] == 'LATTE':
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=forces)
        preprocess.parse_parameters(parameters['descriptor_params']) 
        extra_data = {
            'Rcut': preprocess.gvect['Rc'],
            'species': preprocess.species_idx_2str,
            'type': 'LATTE',
            'nn_mode': 'list',
            'pair_mod': parameters['pair_mod'],
            'at_mod': parameters['at_mod'],
            'pair_max': parameters['pair_max']} # added by Gabriele
    
    if long_range or long_range_pol: # long range params valid for both train and validation sets
        gaussian_charges = parameters['gaussian_charges'][0] != b'False'
        precompute_ewald_sums = parameters['precompute_ewald_sums']#estimate_memory_usage(Nconf, Nats_max, parameters['at_mod'], mode='training')

        extra_data['ewald_input_params'] = parameters['ewald_input_params']
        extra_data['verbose'] = parameters['verbose']
        extra_data['gaussian_charges'] = parameters['gaussian_charges']
        extra_data['BORN'] = parameters['BORN']

        # if precompute_ewald_sums:
        #     # e,f,linsys_masks = compute_ewald_sums(parameters['ewald_input_params'], all_positions, all_latvecs, Nats_list)
        #     extra_data['ewald_params'] = (e,f,linsys_masks) # the kernels just computed
        # else:
        #     extra_data['ewald_params'] = parameters['ewald_input_params'] # to compute Ewald elsewhere
        
        
    if parameters['data_dir']:
        if long_range or long_range_pol:
            # this computes the max amount of atoms for padding
            (Nconf, Nats_max, Nats_list), all_positions, all_latvecs = get_all_positions_and_latvecs(parameters['data_dir'], parameters['at_mod'])
            extra_data['Nats_max'] = Nats_max

            dataset = input_pipeline_lr(
                data_dir=parameters['data_dir'],
                batch_size=parameters['batch_size'],
                # parse_fn=PF,
                forces=forces,
                charges=charges,
                dipoles=dipoles,
                stress=stress,
                num_parallel_calls=parameters['num_parallel_calls'],
                cache=parameters['data_cache'],
                shuffle=True,
                input_format=parameters['input_format'],
                pbc = parameters['pbc'],
                long_range_pol = long_range_pol, # polarisable model
                gaussian_charges=gaussian_charges,
                precompute_ewald_sums = precompute_ewald_sums,
                extra_data=extra_data
                )
        else:
            dataset = input_pipeline(
                data_dir=parameters['data_dir'],
                batch_size=parameters['batch_size'],
                parse_fn=PF,
                stress=stress,
                num_parallel_calls=parameters['num_parallel_calls'],
                cache=parameters['data_cache'],
                shuffle=True,
                input_format=parameters['input_format'],
                pbc = parameters['pbc'],
                extra_data=extra_data)
    else:
        dataset = None
    
    if parameters['val_data_dir']:
        if long_range or long_range_pol:
            # this computes the max amount of atoms for padding
            (Nconf, Nats_max, Nats_list), all_positions, all_latvecs = get_all_positions_and_latvecs(parameters['val_data_dir'], parameters['at_mod'])
            extra_data['Nats_max'] = Nats_max
            
            vdata = input_pipeline_lr(
                data_dir=parameters['val_data_dir'],
                batch_size=parameters['batch_size'],
                # parse_fn=PF,
                forces=forces,
                charges=charges,
                dipoles=dipoles,
                stress=stress,
                num_parallel_calls=parameters['num_parallel_calls'],
                cache=parameters['data_cache'],
                oneshot=True,
                shuffle=False,
                input_format=parameters['input_format'],
                pbc = parameters['pbc'],
                long_range_pol = long_range_pol, # polarisable model
                gaussian_charges=gaussian_charges,
                precompute_ewald_sums = precompute_ewald_sums,
                extra_data=extra_data)
        else:
            vdata = input_pipeline(
                data_dir=parameters['val_data_dir'],
                batch_size=parameters['batch_size'],
                parse_fn=PF,
                stress=stress,
                num_parallel_calls=parameters['num_parallel_calls'],
                cache=parameters['data_cache'],
                oneshot=True,
                shuffle=False,
                input_format=parameters['input_format'],
                extra_data=extra_data)
    else:
        vdata = None
    return dataset, vdata, preprocess


def get_all_positions_and_latvecs(data_dir, at_mod):
    all_examples = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[-1] == '.example':
            all_examples.append(file)
    Nconf = len(all_examples)      
    Nats_list = np.zeros(Nconf, dtype=np.int32)
    all_positions, all_latvecs = np.zeros((Nconf, at_mod, 3), dtype=np.float32), np.zeros((Nconf, 3, 3), dtype=np.float32)
    for i in range(len(all_examples)):   
        # print(os.path.join(data_dir,all_examples[i]))
        example = ExampleJsonWrapper(os.path.join(data_dir,all_examples[i]))
        
        
        this_pos = example.angstrom_positions
        this_cell = example.angstrom_lattice_vectors
        Nats = len(this_pos)
        Nats_list[i] = Nats

        
        this_pos_padded = np.pad(this_pos, ( (0,at_mod-Nats), (0,0) ), constant_values=np.inf)
        all_positions[i] = this_pos_padded
        all_latvecs[i] = this_cell
        
        
        
    Nats_max = Nats_list.max()
    
    return (Nconf, Nats_max, Nats_list), all_positions, all_latvecs


def estimate_memory_usage(Nconf, Nats_max, at_mod, percentage=0.75, gpu=False, mode='training'):
    bytes_to_Gb = 1/1024/1024/1024
    if gpu:
        import nvidia_smi
        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        print("Total memory:", info.total)
        print("Free memory:", info.free)
        print("Used memory:", info.used)

        nvidia_smi.nvmlShutdown()
    
    else:
        avail_memory = psutil.virtual_memory().available * bytes_to_Gb
        
    # inda, indb, nn_r, sp_a, sp_b, mask ~ Nats^2
    # nn_vecs ~ Nats^2 * 3
    # el_kernels ~ at_mod^2 * 4, linsys_masks ~ at_mod^2
    float32 = 4 # bytes
    estimated_memory_usage = Nconf * (9 * Nats_max**2 + 5 * at_mod**2) * float32 * bytes_to_Gb
    
    if estimated_memory_usage < avail_memory * percentage:
        print('INFO - I will store the {} batch in RAM'.format(mode))
        precompute_ewald_sums = True
    
    return precompute_ewald_sums

# def compute_ewald_sums(ewald_inputs, all_positions, all_latvecs, Nats_list):
#     from panna.electrostatics.Electrostatics_qq_jax import create_masks, set_ewald_parameters_for_batch, Ewald_summation
#     import time
    
#     at_mod = len(all_positions[0,:,0])
#     t0 = time.time()
#     # --- Ewald summation ---
#     # unpacking inputs for Ewald sum
#     acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = ewald_inputs
    
#     # alpha = 1/(np.sqrt(2)*gaussian_smearing) if not np.isnan(gaussian_smearing) else np.nan
#     # real_space_cutoff = real_space_cutoff if real_space_cutoff is not np.nan else None
#     # rec_space_cutoff = rec_space_cutoff if rec_space_cutoff is not np.nan else None


#     # input_realspace, alpha, input_recspace = set_ewald_parameters(positions, example.lattice_vectors, acc_factor, Rc=real_space_cutoff, alpha=alpha, Kc_sc=rec_space_cutoff, verbose=False)
#     input_realspace, alphas, input_recspace = set_ewald_parameters_for_batch(all_positions, all_latvecs, acc_factor)


#     ew = jax.jit(jax.vmap(partial(Ewald_summation, input_realspace=input_realspace, input_recspace=input_recspace)))
#     out_ewald = ew(all_positions,
#         all_latvecs,
#         alphas,
#         # input_realspace,
#         # input_recspace,
#         )
#     e = out_ewald['el_energy_kernel']
#     f = out_ewald['el_force_kernel']
#     # b = out_ewald['linsys_bvector']
#     # linsys_masks = out_ewald['linsys_masks']
#     # --------------------
#     t1 = time.time()
#     print(t1-t0, ' sec Ewald sum')
    
    
#     linsys_masks = create_masks(Nats_list, at_mod)



#     return np.array(e),np.array(f), linsys_masks



# def create_dataset(parameters):
#     forces = parameters['forces_cost']>0.0
#     # Dummy parseFn
#     PF = dummyPF(forces)
#     # This works with LATTE for now
#     if parameters['descriptor_type'] == 'LATTE':
#         preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=forces)
#         preprocess.parse_parameters(parameters['descriptor_params'])        
#         extra_data = {
#             'Rcut': preprocess.gvect['Rc'],
#             'species': preprocess.species_idx_2str,
#             'type': 'LATTE',
#             'nn_mode': 'list',
#             'pair_mod': parameters['pair_mod'],
#             'at_mod': parameters['at_mod'],
#             'ewald_input_params' : parameters['ewald_input_params']} # added for LR
#     if parameters['data_dir']:
#         dataset = input_pipeline(
#             data_dir=parameters['data_dir'],
#             batch_size=parameters['batch_size'],
#             parse_fn=PF,
#             num_parallel_calls=parameters['num_parallel_calls'],
#             cache=parameters['data_cache'],
#             shuffle=True,
#             input_format=parameters['input_format'],
#             extra_data=extra_data)
#     else:
#         dataset = None
#     if parameters['val_data_dir']:
#         vdata = input_pipeline(
#             data_dir=parameters['val_data_dir'],
#             batch_size=parameters['batch_size'],
#             parse_fn=PF,
#             num_parallel_calls=parameters['num_parallel_calls'],
#             cache=parameters['data_cache'],
#             oneshot=True,
#             shuffle=False,
#             input_format=parameters['input_format'],
#             extra_data=extra_data)
#     else:
#         vdata = None
#     return dataset, vdata, preprocess


        
        
        

