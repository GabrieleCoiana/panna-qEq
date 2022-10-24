from abc import ABC, abstractmethod


class ScaffoldBasic(ABC):
    """ A recipe for the PANNA basic network

    Parameters
    ----------
    config: a config to extract the quantities needed for the
            networks setup
    name: a name for the network, if more are present
    """
    def __init__(self, config, name):
        self._config = config
        self._name = name
        self._usable_for_training = False
        self._usable_for_validation = False

    @classmethod
    @abstractmethod
    def _scaffold_type(cls):
        """string defining scaffold type
        """

    @property
    @abstractmethod
    def tfr_parse_function(self):
        """Function needed to parse tfr files
        the returned function must be able to take???
        """

    def evaluate(self, example, forces, add_offset):
        """Evaluation in numpy
        This will be deprecated when we will move to eager mode
        """
        raise NotImplementedError

    @abstractmethod
    def tf_network(self, batch_size, batches):
        """Main function to predict the output of the network
        """

    def dump_scaffold(self, folder, extra_data=None):
        """method to load a network from an internal (panna) checkpoint
        (no TF!!)
        """
        self.dump_scaffold_metadata(folder, extra_data=extra_data)
        self.dump_scaffold_data(folder)

    @abstractmethod
    def dump_scaffold_metadata(self, folder, extra_data=None):
        """method to dump scaffold metadata
        (no TF!!)
        """

    @abstractmethod
    def dump_scaffold_data(self, folder):
        """method to dump scaffold metadata
        (no TF!!)
        """

    @abstractmethod
    def load_dump(self, folder):
        """method to serialize the class
        Retruns
        """

    def dump_scaffold_lammps(self, folder, extradata, *, filename='panna.in'):
        """ Implement only if a lammaps plugin is available for
        the scaffold
        """
        raise NotImplementedError
