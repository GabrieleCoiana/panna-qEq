"""Force evaluate engine."""

import itertools
import enum

from typing import Union
from typing import Sequence

import tensorflow as tf


class OpVersion(enum.Enum):
    """Numerical option for the matrix matrix multiplication."""

    SPARSE = 'sparse'
    DENSE = 'dense'


class ForceEvaluate():
    """Class to encapsulate the logic needed to compute the force.

    Parameters
    ----------
    g_size:
        size of the descriptor,
        this information is needed to perform some reshapes
    dg_dx_format_sparse:
        dg_dx can be stored as
        - sparse matrix for large cell or as
        - dense matrix for small cells
        this tell the code how the information will be passed.
        The dataset must have uniform shaping.
    op_version:
        Still a WIP, if we want to perform a sparse multiplication
        or a dense multiplication.
        if dg_dx_format_sparse is False then only dense is available.
    """
    def __init__(self,
                 g_size: int,
                 dg_dx_format_sparse: bool = True,
                 op_version: Union[OpVersion, str] = OpVersion.SPARSE):
        self._g_size = g_size
        self._dg_dx_format_sparse = dg_dx_format_sparse
        self._op_version = OpVersion(op_version)

    def _adapt_input(self, dg_dx_info):
        def _input_to_tensor(x: Sequence[Sequence[Union[int, float]]],
                             dtype=None) -> tf.RaggedTensor:
            # Can this be done more efficiently without ragged?
            lens = [len(row) for row in x]
            maxl = tf.reduce_max(lens)
            tens = [tf.concat([row,tf.zeros(maxl-len(row))], axis=0) for row in x]
            tens = tf.stack(tens)
            if dtype:
                tens = tf.cast(tens, dtype=dtype)
            return tens

        dg_dx_info_new = []

        if isinstance(dg_dx_info[0], Sequence):
            dg_dx_info_new.append(_input_to_tensor(dg_dx_info[0], dtype=tf.float32))
        else:
            dg_dx_info_new.append(dg_dx_info[0])

        if self._dg_dx_format_sparse:
            if isinstance(dg_dx_info[1], Sequence):
                dg_dx_info_new.append(_input_to_tensor(dg_dx_info[1], dtype=tf.float32))
            else:
                dg_dx_info_new.append(dg_dx_info[1])

            if isinstance(dg_dx_info[2], Sequence):
                dg_dx_info_new.append(_input_to_tensor(dg_dx_info[2], dtype=tf.float32))
            else:
                dg_dx_info_new.append(dg_dx_info[2])

        return tuple(dg_dx_info_new)

    def _create_dg_dx_sparse_tensor(self, n_atoms, dg_dx_v, dg_dx_i1, dg_dx_i2):
        dg_dx_i1 = tf.cast(dg_dx_i1, dtype=tf.int64)
        dg_dx_i2 = tf.cast(dg_dx_i2, dtype=tf.int64)
        indices = tf.stack([dg_dx_i1, dg_dx_i2], axis=1)
        dg_dx = tf.SparseTensor(indices=indices,
                                values=dg_dx_v,
                                dense_shape=[n_atoms * self._g_size, n_atoms * 3])
        return dg_dx

    # def _create_dg_dx_dense_tensor(self, n_atoms, dg_dx):
    #     dg_dx = tf.reshape(dg_dx, [n_atoms * self._g_size, n_atoms * 3])
    #     return dg_dx

    def _compute_per_cell_force_gsparse(self, x):
        r"""Operating on each element of the batch.

        x contains: read the code, self explanatory

        Elements are sliced (if padded) and reshaped to compute

        .. math:

           F_k = \sum_{ij} dE/dg_{ij} dg{ij}/dx_k

        """
        n_atoms = tf.cast(x[1][0],tf.int32)
        atmax = tf.shape(x[0])[0]
        de_dg = tf.reshape(x[0][:n_atoms],[-1])

        dg_dx_v, dg_dx_i1, dg_dx_i2 = x[2], x[3], x[4]
        dg_dx = self._create_dg_dx_sparse_tensor(n_atoms, dg_dx_v, dg_dx_i1,
                                                dg_dx_i2)
        if self._op_version == OpVersion.SPARSE:
            de_dg = tf.reshape(de_dg, [1, -1])
            forces_pred = -tf.reshape(tf.sparse.sparse_dense_matmul(de_dg, dg_dx),
                                     [-1])
        elif self._op_version == OpVersion.DENSE:
            print("Sparse to dense not implemented!")
            # de_dg = tf.reshape(de_dg, -1)
            # dg_dx = tf.sparse.to_dense(dg_dx)
            # forces_pred = -tf.linalg.matvec(
            #     dg_dx, de_dg, transpose_a=True, a_is_sparse=True)
        else:
            raise ValueError(f'{self._op_version} not supported')            
        forces_pred = tf.concat([forces_pred,tf.zeros((atmax-n_atoms)*3)], 0)
        return forces_pred

    def _compute_per_cell_force_gdense(self, x):
        r"""Operating on each element of the batch.

        x contains: read the code, self explanatory

        Elements are sliced (if padded) and reshaped to compute

        .. math:

           F_k = \sum_{ij} dE/dg_{ij} dg{ij}/dx_k

        """
        n_atoms = tf.cast(x[1][0],tf.int32)
        atmax = tf.shape(x[0])[0]
        de_dg = tf.reshape(x[0][:n_atoms], [-1])
        dg_dx = tf.reshape(x[2][:n_atoms**2*self._g_size*3], 
                           [n_atoms * self._g_size, n_atoms * 3])
        if self._op_version == OpVersion.DENSE:
            forces_pred = -tf.linalg.matvec(
                dg_dx, de_dg, transpose_a=True, a_is_sparse=False)
        elif self._op_version == OpVersion.SPARSE:
            raise ValueError('dg_dx matrix is dense, sparse op not possible')
        else:
            raise ValueError(f'{self._op_version} not supported')
        forces_pred = tf.concat([forces_pred,tf.zeros((atmax-n_atoms)*3)], 0)

        return forces_pred

    def __call__(self, batch_of_des_dgs, batch_n_atoms, dg_dx_info, training=None):
        """Compute the forces on each atom.

        Returnif self._dg_dx_format_sparse:
        ------
        tf tensor of shape [batch_size, n_atoms_max, 3]
        """
        if not training:
            self._adapt_input(dg_dx_info)

        if self._dg_dx_format_sparse:
            batch_n_atoms = tf.cast(tf.reshape(batch_n_atoms, [-1, 1]), dtype=tf.float32)
            compute_stack = (batch_of_des_dgs, batch_n_atoms,
                             dg_dx_info[0], dg_dx_info[1], dg_dx_info[2])
            output_signature = tf.TensorSpec(shape=[None], dtype=tf.float32)
            return tf.map_fn(self._compute_per_cell_force_gsparse,
                             compute_stack,
                             fn_output_signature=output_signature)
        else:
            batch_n_atoms = tf.cast(tf.reshape(batch_n_atoms, [-1, 1]), dtype=tf.float32)
            compute_stack = (batch_of_des_dgs, batch_n_atoms, dg_dx_info[0])
            output_signature = tf.TensorSpec(shape=[None], dtype=tf.float32)
            return tf.map_fn(self._compute_per_cell_force_gdense,
                             compute_stack,
                             fn_output_signature=output_signature)

    def debug_call(self, _, batch_n_atoms, dg_dx_info, training=None):
        """To exclude the actual force calculation.

        This call is compatible with __call__ and does not preform any force logic.

        Return
        ------
          tensor of shape [batch_size, n_atoms_max, 3] and full of zeros.
        """
        batch_size = tf.shape(batch_n_atoms)[0]
        atoms_max = tf.reduce_max(batch_n_atoms)
        forces = tf.zeros([batch_size, atoms_max, 3])
        return forces
