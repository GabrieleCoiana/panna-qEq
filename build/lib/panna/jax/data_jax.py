from panna.neuralnet.inputs_iterator import input_pipeline
from panna.gvector import GvectLATTE

# Dummy class since we don't need the complete ParseFn
class dummyPF():
    def __init__(self,
                 forces: bool = False):
        self._forces = forces

    def __call__(self):
        return None

def create_dataset(parameters):
    forces = parameters['forces']
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
            'pair_mod': parameters['pair_mod'],
            'at_mod': parameters['at_mod']}
    if parameters['data_dir']:
        dataset = input_pipeline(
            data_dir=parameters['data_dir'],
            batch_size=parameters['batch_size'],
            parse_fn=PF,
            num_parallel_calls=parameters['num_parallel_calls'],
            cache=parameters['data_cache'],
            shuffle=True,
            input_format=parameters['input_format'],
            extra_data=extra_data)
    else:
        dataset = None
    if parameters['val_data_dir']:
        vdata = input_pipeline(
            data_dir=parameters['val_data_dir'],
            batch_size=parameters['batch_size'],
            parse_fn=PF,
            num_parallel_calls=parameters['num_parallel_calls'],
            cache=parameters['data_cache'],
            oneshot=True,
            shuffle=False,
            input_format=parameters['input_format'],
            extra_data=extra_data)
    else:
        vdata = None
    return dataset, vdata, preprocess

