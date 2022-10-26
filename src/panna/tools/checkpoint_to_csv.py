from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event
import pandas as pd

import logging
import os
import argparse

logger = logging.getLogger(__name__)


def _load_events(folder):
    """ load events form a folder
    """
    logger.info('loading ckpts, this op can take a while')
    events = EventAccumulator(folder).Reload()
    return events


def _extract_scalars(events):
    """ extract scalars form events
    """
    scalars = events.scalars
    return zip(scalars.Keys(), [scalars.Items(key) for key in scalars.Keys()])


def _scalar_to_pd_df(list_scalar):
    """ convert a list of scalar obj to pd DataFrame
    """
    tmp = []
    for scalar in list_scalar:
        tmp.append((scalar.wall_time, scalar.step, scalar.value))
    df = pd.DataFrame(tmp, columns=['wall_time', 'step', 'value'])
    return df


def extract(indir, outdir):
    # create outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    events = _load_events(indir)
    scalars = _extract_scalars(events)
    for key, list_scalars in scalars:
        logger.info('parsing saving %s', key)
        dataframe = _scalar_to_pd_df(list_scalars)
        dataframe.to_csv(os.path.join(outdir, key.replace('/', '_') + '.csv'))


def main():
    parser = argparse.ArgumentParser(description='update bin example files')
    parser.add_argument('-i',
                        '--indir',
                        type=str,
                        help='in path',
                        default='checkpoints')
    parser.add_argument('-o',
                        '--outdir',
                        default='checkpoints_csv',
                        type=str,
                        help='out path')
    args = parser.parse_args()
    extract(indir=args.indir, outdir=args.outdir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s - %(message)s')
    main()
