"""Parser for the validation config file."""

import logging

from panna.neuralnet.config_parser_train import parse_network, ModelParams, example_param_parser
from panna.neuralnet.inputs_iterator import input_pipeline
from panna.neuralnet.parse_fn import ParseFn
from panna.neuralnet.checkpoint import recover_models_files

logger = logging.getLogger(__name__)


def _filter_models_files(train_dir, validation_parameters):
    single_step = validation_parameters.getboolean('single_step', False)
    epoch_number = validation_parameters.getint('epoch_number', None)
    step_number = validation_parameters.getint('step_number', None)
    subsampling = validation_parameters.getint('subsampling', None)

    model_files = recover_models_files(train_dir)
    start = validation_parameters.getint('start', 0)
    end = validation_parameters.getint('end', len(model_files))

    if (single_step and not step_number):
        logger.info('evaluating last checkpoint')
        model_files = [model_files[-1]]
    elif (single_step and step_number and epoch_number):
        logger.info('evaluating batch number %d, step number %d', epoch_number,
                    step_number)
        for eg in model_files:
            if eg == (epoch_number, step_number):
                model_files = [eg]
                break
        else:
            raise ValueError('epoch %d step %d not found', epoch_number, step_number)

    elif (not single_step and subsampling):
        logger.info('evaluation of all the checkpoints at steps of %d ', subsampling)
        model_files = model_files[start:end:subsampling]
    else:
        logger.info('evaluation of all the checkpoints in range %d, %d', start, end)
        logger.info('range is no in steps but files for now')
        model_files = model_files[start:end]

    return model_files


def _create_validation_dataset(io_info, validation_info, parse_fn):
    data_dir = io_info.get('data_dir')
    batch_size = validation_info.getint('batch_size', 1)
    input_format = io_info.get('input_format', 'tfr')
    if input_format == 'example':
        preprocess, extra_data = example_param_parser(io_info.get('gvect_ini', None),
                                                      parse_fn._forces)
    elif input_format == 'tfr':
        preprocess = None
        extra_data = {}
    else:
        raise ValueError('Input format not supported')
    if not data_dir:
        raise ValueError('Missing data directory')
    dataset = input_pipeline(data_dir=data_dir,
                             batch_size=batch_size,
                             parse_fn=parse_fn,
                             oneshot=True,
                             input_format=input_format,
                             extra_data=extra_data)
    return dataset, preprocess


def parse_validation_config(valid_config, train_config):
    """Validation parser call."""
    _ = valid_config['PARALLELIZATION']
    validation_options = valid_config['VALIDATION_OPTIONS']
    compute_forces = validation_options.getboolean('compute_forces', False)
    io_info = valid_config['IO_INFORMATION']

    if valid_config.has_section('LONG_RANGE'):
        LR_info = valid_config['LONG_RANGE']
        long_range_el = LR_info.getboolean('long_range_el', False)
        long_range_vdw = LR_info.getboolean('long_range_vdw', False)
    else:
        long_range_el = False
        long_range_vdw = False


    networks_dir = io_info.get('networks_dir')
    model_files = _filter_models_files(networks_dir, validation_options)

    # Neural Network and data
    default_nn_config, networks_config, data_info = parse_network(train_config)

    # final touch:
    default_nn_config[1].compute_jacobian = compute_forces
    for _, nn_config in networks_config:
        if compute_forces:
            nn_config.compute_jacobian = True

    parse_fn = ParseFn(g_size=networks_config[0][1].feature_size,
                       n_species=data_info.n_species,
                       forces=compute_forces,
                       sparse_dgvect=data_info.sparse_derivatives,
                       energy_rescale=data_info.energy_rescale,
                       names=data_info.examples_name,
                       long_range_el=long_range_el)

    g_size = networks_config[0][1].feature_size

    # Copying the input format from train, or overriding
    if 'input_format' not in io_info:
        io_info['input_format'] = train_config['IO_INFORMATION'].get('input_format', 'tfr')
    if io_info['input_format'] == 'example':
        if 'gvect_ini' not in io_info:
            if 'gvect_ini' in train_config['IO_INFORMATION']:
                io_info['gvect_ini'] = train_config['IO_INFORMATION'].get('gvect_ini', None)
            else:
                raise ValueError('Gvector information not found.')
    dataset, preprocess = _create_validation_dataset(io_info, 
                                                     validation_options, parse_fn)

    # batch size is needed in the model because of some reshapes
    model_params = ModelParams(default_nn_config,
                               networks_config,
                               g_size,
                               data_info.sparse_derivatives,
                               compute_forces,
                               examples_name=data_info.examples_name,
                               input_format=io_info['input_format'],
                               preprocess=preprocess,
                               max_atoms=data_info.max_atoms,
                               long_range_el=long_range_el)

    return model_params, model_files, dataset

