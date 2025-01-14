import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as partial


# Class to handle preprocessing with trainable parameters
class HKpre(hk.Module):
    def __init__(self, pre, parameters):
        super().__init__()
        self.dfunc = pre.jax_g
        self.dfunc_postg = pre.jax_g_sum
        self.varinfo = pre.var_info_jax()
        # If species cutoff is requested, we build here the matrix updating terms
        if parameters['sp_cutoff']:
            spl = parameters['species']
            nsp = len(spl)
            sp_Rc = parameters['cutoff']*np.ones((nsp,nsp,1))
            for s in parameters['sp_cutoff'].split(':'):
                sp1, sp2, rc = [t.strip() for t in s.split(',')]
                sp_Rc[spl.index(sp1),spl.index(sp2)] = rc
        parameters['clipping_vars'] = []
        if not parameters['fixed_descr']:
            # Here we create a list of parameters to be clipped
            # We report name1 and name2 to identify the variable to be clipped
            # We have different types of clipping:
            # type1 has structure: [1, name1, name2, min, max]
            #       and will keep name1.name2 in [min, max]
            # type2 has structure: [2, name1, name2, name3, max]
            #       and will keep name1.name2 in [0, max-name1.name3]
            #       (on a per-element basis)
            # type3 has structure: [3, name1, name2, name3, name4, max_matrix]
            #       where max_matrix has [nsp, nsp] shape
            #       and will set name1.name2 to 0 if name1.name3+name1.name4 > max_matrix
            #       (on a per element and per species basis)
            for i,v in enumerate(self.varinfo):
                # If sigma is const, we just set min/max for clipping
                if not parameters['learnsig']:
                    if v['name'].split('_')[0]=='cent':
                        parameters['clipping_vars'].append([1, 'panna_mlp/~/h_kpre',
                                   'DV_'+v['name'],v['min'],v['max']])
                # Otherwise, we set centers to depend on sigma, and sigma to its limits
                else:
                    if v['name'].split('_')[0]=='cent':
                        parameters['clipping_vars'].append([2, 'panna_mlp/~/h_kpre',
                                   'DV_'+v['name'],'DV_'+v['name'].replace('cent','sig'),parameters['cutoff']])
                    if v['name'].split('_')[0]=='sig':
                        parameters['clipping_vars'].append([1, 'panna_mlp/~/h_kpre',
                                   'DV_'+v['name'],v['min'],v['max']])
                    if parameters['sp_cutoff']:
                        if v['name'].split('_')[0]=='spw':
                            parameters['clipping_vars'].append([3, 'panna_mlp/~/h_kpre',
                                       'DV_'+v['name'],'DV_'+v['name'].replace('spw','cent'),
                                       'DV_'+v['name'].replace('spw','sig'), sp_Rc])
                        

        # Handling species embedding
        if parameters['sp_emb_file']:
            self.sp_emb_size = parameters['sp_emb_size']
            self.sp_embedding = jnp.asarray(parameters['sp_embedding'])
        else:
            self.sp_embedding = None

        self.dsize = pre.gsize
        self.deriv = parameters['forces']

    def __call__(self, nn_vecs, inputs):
        dvars = []
        for i,v in enumerate(self.varinfo):
            if v['type']=='norm':
                dvars.append(hk.get_parameter('DV_'+v['name'],
                    v['shape'],init=hk.initializers.RandomNormal(v['std'],v['mean'])))
            elif v['type']=='unif':
                dvars.append(hk.get_parameter('DV_'+v['name'],
                    v['shape'],init=hk.initializers.RandomUniform(v['min'],v['max'])))
        if self.sp_embedding is not None:
            dvars.append(self.sp_embedding)

        # We call the function first to set the params, then vmap over the inputs
        dwithvars = partial(self.dfunc, dvars)
        modinputs = {'nn_vecs':nn_vecs,
                     'nn_r':jnp.sqrt(jnp.sum(nn_vecs**2,axis=-1)),
                     'sp_a':inputs['sp_a'],
                     'sp_b':inputs['sp_b'],
                     }
        # We vmap over pairs, obtain dicts of g and dg contributions from that pair
        gpairs = jax.vmap(dwithvars, in_axes=({i:0 for i in modinputs},))(modinputs)
        nats = jnp.shape(inputs['inde'])[0]

        gpatom = {}
        for gpk in gpairs.keys():
            up_dim = list(jnp.shape(gpairs[gpk]))[1:]
            up_tuple = tuple(np.arange(1,len(up_dim)+1))
            dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=up_tuple, 
                                                      inserted_window_dims=(0,),
                                                      scatter_dims_to_operand_dims=(0,))
            gpatom[gpk] = jax.lax.scatter_add(jnp.zeros([nats]+up_dim),
                                jnp.expand_dims(inputs['inda'],-1),gpairs[gpk],
                                dimnums,mode='drop')

        # We vmap over atoms, get g contribution for that atom
        g = jax.vmap(self.dfunc_postg, 
                     in_axes=({i:0 for i in gpatom},))(gpatom)
        return g

class atom_Linear(hk.Module):
    def __init__(self, nsp, insize, outsize, act, sp_w, sp_emb=None):
        super().__init__()
        self.nsp = nsp
        self.insize = insize
        self.outsize = outsize
        self.sp_w = sp_w
        if sp_w=='e':
            self.emb = jnp.asarray(sp_emb)
            self.emb_size = sp_emb.shape[1]
        if act=='lin':
            self.act = None
        elif act=='exp':
            self.act = lambda x: jnp.exp(-x*x)
        elif act=='tanh':
            self.act = jnp.tanh
        elif act=='silu':
            self.act = jax.nn.silu
        elif act=='lorentz':
            self.act = lambda x: 1./(1.+x*x)
        else:
            raise ValueError('Unknown activation')

    def __call__(self, species, inputs):
        if self.sp_w=='s':
            # Weights for each species, Glorot initialization
            w = hk.get_parameter('w',(self.nsp,self.outsize,self.insize),
                    init=hk.initializers.RandomNormal(jnp.sqrt(2./(self.insize+self.outsize)),0.0))
            b = hk.get_parameter('b',(self.nsp,self.outsize),init=hk.initializers.Constant(0.0))
            out = jnp.matmul(w[species],inputs) + b[species]
        elif self.sp_w=='c':
            # Not selecting for species
            w = hk.get_parameter('w',(self.outsize,self.insize),
                    init=hk.initializers.RandomNormal(jnp.sqrt(2./(self.insize+self.outsize)),0.0))
            b = hk.get_parameter('b',(self.outsize,),init=hk.initializers.Constant(0.0))
            out = jnp.matmul(w,inputs) + b
        elif self.sp_w=='e':
            # Using second dim for the embedding
            w = hk.get_parameter('w',(self.outsize,self.emb_size,self.insize),
                    init=hk.initializers.RandomNormal(jnp.sqrt(2./(self.insize+self.outsize)),0.0))
            b = hk.get_parameter('b',(self.emb_size,self.outsize,),init=hk.initializers.Constant(0.0))
            out = jnp.matmul(jnp.matmul(self.emb[species],w),inputs) + jnp.matmul(self.emb[species],b)
        if self.act:
            out = self.act(out)
        return out

def E_func(x, sp, layers, t=0):
    for i,l in enumerate(layers):
        x = l(sp,x)
    return x[t]

def pairs2E(disp, batch, dfunc, layers, multitarget=False):
    des = dfunc(disp, batch)
    if multitarget:
        Ei = jax.vmap(E_func, in_axes=(0,0,None,0))(des,batch['species'],
                                                    layers,batch['targets'])
    else:
        Ei = jax.vmap(E_func, in_axes=(0,0,None))(des,batch['species'],
                                                  layers)
    return jnp.sum(Ei), Ei

# Species resolved MLP
# Handling pairs of neighs as input
class PANNA_MLP(hk.Module):
    def __init__(self, parameters, pre, **kwargs):
        super().__init__()
        self.name = 'panna_mlp'
        # Disabling derivative in the gvect function
        pre.compute_dgvect = False
        self.dfunc = HKpre(pre, parameters)
        self.arch = [self.dfunc.dsize]+parameters['mlp_arch']
        self.nsp = len(parameters['species'])
        self.layers = []
        self.offsets = jnp.array(parameters['offsets'])
        self.forces = parameters['forces']
        # To handle species specific/common mlp weights
        sp_weights = parameters['mlp_sp_weights']
        if sp_weights:
            sp_weights = [s.strip() for s in sp_weights.split(':')]
        else:
            sp_weights = ['s']*len(self.arch)
        # To handle frozen layers
        trainable = parameters['mlp_trainable']
        if trainable:
            trainable = [{'T':True, 'F':False}[t.strip()] for t in trainable.split(':')]
        else:
            trainable = [True]*len(self.arch)

        # Special case if we just need the descriptor
        self.descr_only = parameters['task']=='descr'

        # Handling different possible targets (datasets)
        self.num_targets = parameters['num_targets']
        if self.num_targets>1:
            # We take care of multiple targets internally
            self.arch[-1] *= self.num_targets
            self.multitarget = True
        else:
            self.multitarget = False

        for i in range(len(self.arch)-1):
            if i<len(self.arch)-2:
                act = parameters['mlp_act']
            else:
                # Last layer
                act = 'lin'
            self.layers.append(atom_Linear(self.nsp,self.arch[i],self.arch[i+1],act,
                               sp_weights[i],parameters['sp_embedding']))
            if not trainable[i]:
                fixvar = self.name+'/~/'+self.layers[-1].name
                if fixvar not in parameters['fixed_variables']:
                    parameters['fixed_variables'].append(fixvar)


    def __call__(self, batch):
        if self.descr_only:
            return self.dfunc(batch['nn_vecs'], batch)

        if self.forces:
            # Computing directly E and dE/dpair
            (Esum, Ei), Fterms = jax.value_and_grad(pairs2E, has_aux=True)(
                                 batch['nn_vecs'], batch, self.dfunc, self.layers,
                                 self.multitarget)
        else:
            Esum, Ei = pairs2E(batch['nn_vecs'], batch, self.dfunc, self.layers,
                               self.multitarget)
        nats = jnp.shape(batch['species'])[0]
        bsize = jnp.shape(batch['nats'])[0]

        if self.multitarget:
           Ei += self.offsets[batch['targets'],batch['species']]
        else:
           Ei += self.offsets[batch['species']]
        # Summing the right elements (indices inde in the batch)
        dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), 
                                                  inserted_window_dims=(0,),
                                                  scatter_dims_to_operand_dims=(0,))
        E = jax.lax.scatter_add(jnp.zeros(bsize),jnp.expand_dims(batch['inde'],-1),Ei,
                                dimnums,mode='drop')

        if not self.forces:
            return [E]

        dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(1,), 
                                                  inserted_window_dims=(0,),
                                                  scatter_dims_to_operand_dims=(0,))
        # Gathering forces on atoms a of pairs
        F = jax.lax.scatter_add(jnp.zeros((nats,3)),
                                jnp.expand_dims(batch['inda'],-1),Fterms,
                                dimnums,mode='drop')
        # Gathering forces on atoms b of pairs, with other sign
        F -= jax.lax.scatter_add(jnp.zeros((nats,3)),
                                 jnp.expand_dims(batch['indb'],-1),Fterms,
                                 dimnums,mode='drop')
        return [E, F]
