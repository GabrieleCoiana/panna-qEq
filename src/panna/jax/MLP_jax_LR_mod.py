import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as partial
# point charges
from panna.electrostatics.Electrostatics_jax_qq import Ewald_summation, recewald_qq_jax_vect_gaussians, Electrostatics_finite_systems_gaussians
from panna.electrostatics.Electrostatics_jax_qq import Electrostatics_finite_systems
# Gaussian charges
from panna.electrostatics.Electrostatics_jax_qq_gc import Ewald_summation as ewald_gc
from mendeleev import element
import time
from .neural_network import HKpre, atom_Linear, gather_forces

# Species resolved MLP
# Handling pairs of neighs as input
class PANNA_MLP_LR_mod(hk.Module):
    def __init__(self, parameters, pre, **kwargs):
        super().__init__()
        self.name = 'panna_mlp_lr_mod'
        # Disabling derivative in the gvect function
        pre.compute_dgvect = False
        self.dfunc = HKpre(pre, parameters, self.name)
        self.arch = [self.dfunc.dsize]+parameters['mlp_arch']
        self.nsp = len(parameters['species'])
        self.layers = []
        self.offsets = jnp.array(parameters['offsets'])
        self.forces = parameters['forces_cost']>0.0
        self.are_el_kernels_precomputed = parameters['precompute_ewald_sums']
        self.pbc = parameters['pbc']
        # self.phi_ext = parameters['phi_ext'] if 'phi_ext' in parameters else 0.0
        # self.Efield_ext = parameters['Efield_ext'] if 'Efield_ext' in parameters else 0.0
        # self.is_Efield_applied = isinstance(self.Efield_ext, np.ndarray) # True or False
        self.gaussian_charges = parameters['gaussian_charges'][0] not in [ b'False', 'False']
        self.verbose = parameters['verbose']
        self.Zref = jnp.array(parameters['Zed_ref'])
        self.J0_offsets = np.array(parameters['J0_offsets'])
        self.chi0_offsets = np.array(parameters['chi0_offsets'])        
        if np.any(np.isnan(self.J0_offsets)):
            chi0_offsets, self.J0_offsets = self.get_atomic_electronegativity_and_hardness(parameters['species'])
        if np.any(np.isnan(self.chi0_offsets)):
            self.chi0_offsets, J0_offsets = self.get_atomic_electronegativity_and_hardness(parameters['species'])
        self.J0_offsets, self.chi0_offsets = jnp.array(self.J0_offsets), jnp.array(self.chi0_offsets)
        
        

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

        for i in range(len(self.arch)-1):
            if i<len(self.arch)-2:
                act = parameters['mlp_act']
            else:
                act = 'lin'
            self.layers.append(atom_Linear(self.nsp,self.arch[i],self.arch[i+1],act,
                               sp_weights[i],parameters['sp_embedding']))
            if not trainable[i]:
                fixvar = self.name+'/~/'+self.layers[-1].name
                if fixvar not in parameters['fixed_variables']:
                    parameters['fixed_variables'].append(fixvar)
    
    def get_atomic_electronegativity_and_hardness(self, species):
        IP = jnp.array([element(sym).ionenergies[2] for sym in species])
        EA = jnp.array([0.0 if not element(sym).electron_affinity else \
                element(sym).electron_affinity for sym in species])

        J = IP - EA
        chi = 0.5 * (IP + EA)
        return chi, J #previously you had a minus

    def __call__(self, batch):
        # some variables
        nats = jnp.shape(batch['species'])[0]
        bsize = jnp.shape(batch['nats'])[0]
        # real_nats = batch['nats']
        real_nats_tot = jnp.shape(batch['indf'])[0]
        # Compound indices for correct placement of E^2
        inde = batch['inde'][0:real_nats_tot]
        indf = batch['indf']
        dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,),scatter_dims_to_operand_dims=(0,))
        operand = jnp.zeros((nats,3)) # shape: (Natspad, 3)
        inda = jnp.expand_dims(batch['inda'],-1) # shape: (Npairs, 1)
        indb = jnp.expand_dims(batch['indb'],-1) # shape: (Npairs, 1)
        is_Efield_applied = True if len(batch['Efield_ext']) > 0 else False
        Efield_ext = batch['Efield_ext'] if is_Efield_applied else 0.0
        phi_ext = batch['phi_ext'] if is_Efield_applied else 0.0
        extra = {}
                
        if self.descr_only:
            return self.dfunc(batch['nn_vecs'], batch)

        if self.forces:
            # Computing directly E and dE/dpair
            # (Esum, Ei), Fterms = jax.value_and_grad(pairs2E, has_aux=True)(
            #                      batch['nn_vecs'], batch, self.dfunc, self.layers)
            (Esum0, Ei0), Fterms0 = jax.value_and_grad(pairs2E, has_aux=True)(
                                  batch['nn_vecs'], batch, self.dfunc, self.layers, i=0) # shape Fterms: (Npairs, 3)
            Ei1, Ei1_pullback = jax.vjp(
                                        lambda disp: pairs2Ei(
                                            disp,
                                            batch,
                                            self.dfunc,
                                            self.layers,
                                            i=1,
                                        ),
                                        batch['nn_vecs'],
                                    )
            Ei2, Ei2_pullback = jax.vjp(
                                        lambda disp: pairs2Ei(
                                            disp,
                                            batch,
                                            self.dfunc,
                                            self.layers,
                                            i=2,
                                        ),
                                        batch['nn_vecs'],
                                    )
        else:
            (Esum0, Ei0) = pairs2E(batch['nn_vecs'], batch, self.dfunc, self.layers, i=0)
            (Esum1, Ei1) = pairs2E(batch['nn_vecs'], batch, self.dfunc, self.layers, i=1)
            (Esum2, Ei2) = pairs2E(batch['nn_vecs'], batch, self.dfunc, self.layers, i=2)
        

        # offsetting energies, electronegativities and hardnesses
        Ei0 += self.offsets[batch['species']]       # shape: (Natspad,)
        Ei1 += self.chi0_offsets[batch['species']]  # shape: (Natspad,)
        Ei2 += self.J0_offsets[batch['species']]   # shape: (Natspad,)
        
        
        # Summing the right elements (indices inde in the batch)
        # pure short range energy
        E0 = jax.lax.scatter_add(jnp.zeros(bsize),jnp.expand_dims(batch['inde'],-1),Ei0,
                                dimnums,mode='drop') # shape: (bsize,)
        indices = jnp.arange(nats)
        chi = jax.lax.scatter(jnp.zeros(real_nats_tot),indices[:,None],Ei1,
                                dimnums,mode='drop') # shape: (real_nats_tot,)
        J = jax.lax.scatter(jnp.zeros(real_nats_tot),indices[:,None],Ei2,
                                dimnums,mode='drop') # shape: (real_nats_tot,)
        
        if self.are_el_kernels_precomputed: # then take them from batch
            e = jnp.array(batch['el_energy_kernels']) # shape: (bsize, Nmaxats, Nmaxats)
            # f = jnp.array(batch['el_force_kernels'])  # shape: (bsize, Nmaxats, Nmaxats, 3)
        
        else: # you compute them here
            # --- Ewald summation ---
            coords = batch['pos']      # shape: (bsize, Nmaxats, 3)
            latvecs = batch['latvecs'] # shape: (bsize, 3, 3)
            gauss_radii = batch['gaussian_radii'] # shape: (bsize, Nmaxats)
            # input parameters for Ewald sum
            Gcuts, rcuts, input_realspace, alphas_screen, input_recspace = batch['ewald_input_Gcuts'], batch['ewald_input_rcuts'],batch['ewald_input_realspace'],batch['ewald_input_alphas'],batch['ewald_input_recspace']
            
            if self.pbc: 
                if not self.gaussian_charges: # then it's point charges
                    # V = jnp.array([Ewald_summation(coords[i], latvecs[i])['el_energy_kernel'] for i in range(bsize)])
                    ew = jax.jit(jax.vmap(partial(Ewald_summation, input_realspace=input_realspace, input_recspace=input_recspace, verbose=self.verbose)))
                    out_ewald = ew(coords,
                                   latvecs,
                                   alphas_screen,
                                   rcuts,
                                   Gcuts
                                   
                        # input_realspace,
                        # input_recspace,
                        )
                    
                    # compile Ewald for force & stress calculation, for later
                    ew_fs = jax.vmap(partial(Ewald_summation, 
                                          input_realspace=input_realspace, 
                                          input_recspace=input_recspace, 
                                          mode = 'force_stress',
                                          # realmode='vect' if len(batch['pos'])!=1 else 'NN',
                                          # rijs_provided = False if 'ewald_prep' not in batch.keys() else batch['ewald_prep']
                                          ))
                    # --- end Ewald ---
                else: # it's gaussian charges
                    # ew = jax.jit(jax.vmap(partial(recewald_qq_jax_vect_gaussians, input_recspace=input_recspace )))
                    # alphas = 1/(np.sqrt(2) * gauss_radii)
                    # e, f = ew(alphas,coords,latvecs)
                    
                    ew = jax.vmap(partial(ewald_gc, input_realspace=input_realspace, input_recspace=input_recspace ))
                    out_ewald = ew(coords,latvecs,gauss_radii,alphas_screen,rcuts, Gcuts)
                    
                    # compile Ewald for force & stress calculation, for later
                    ew_fs = jax.vmap(partial(ewald_gc, 
                                          input_realspace=input_realspace, 
                                          input_recspace=input_recspace, 
                                          mode = 'force_stress',
                                          realmode='vect' if len(batch['pos'])!=1 else 'NN',
                                          rijs_provided = False if 'ewald_prep' not in batch.keys() else batch['ewald_prep']
                                          ))
                    # --- end Ewald Gaussian ---
                    extra['gauss_radii'] = gauss_radii[:,0:-1]
                    
                e = out_ewald['el_energy_kernel'] # shape: (bsize, Nmaxats, Nmaxats)
                # f = out_ewald['el_force_kernel']  # shape: (bsize, Nmaxats, Nmaxats, 3)
                    
                    
            else:  # TODO: this is now not maintained!
                if not self.gaussian_charges: # compute electrostatics of finite system of point charges
                    el = jax.jit(jax.vmap(Electrostatics_finite_systems))(coords) 
                    e = el['el_energy_kernel'] # shape: (bsize, Nmaxats, Nmaxats)
                    f = el['el_force_kernel']  # shape: (bsize, Nmaxats, Nmaxats, 3)
                else: # compute electrostatics of finite system of Gaussian charges
                    alphas = 1/(np.sqrt(2) * gauss_radii)
                    out = jax.vmap(Electrostatics_finite_systems_gaussians)(coords, alphas)
                    e = out['el_energy_kernel'] # shape: (bsize, Nmaxats, Nmaxats)
                    f = out['el_force_kernel']  # shape: (bsize, Nmaxats, Nmaxats, 3)
            # --------------------
            
        # --- Preparing the linear system ---
        N_linsys = jnp.shape(e)[1] # Nmaxats
        # Compound indices for correct placement of E^2
        compind_mat = (inde, indf, indf) # shape: (real_nats_tot, real_nats_tot, real_nats_tot)
        # Compound indices for correct placement of E^1
        compind_vec = (inde, indf) # shape: (real_nats_tot, real_nats_tot)
        
        # Add hardnesses in diagonals
        e_plus_J = e.at[compind_mat].add(J)  # shape: (bsize, Nmaxats, Nmaxats)
        
        # Add Zi / R_{ij} terms from the \sum_{ij} qi Zj / Rij terms
        Zref = self.Zref[batch['species']][0:real_nats_tot] # shape: (real_nats_tot) 
        Z = jnp.zeros((bsize, N_linsys))    # shape: (bsize, Nmaxats)
        Z = Z.at[compind_vec].set(Zref)     # shape: (bsize, Nmaxats)
        # eQ = e * Z[...,None]                # shape: (bsize, Nmaxats, Nmaxats)
        

        # Apply mask to El kernel
        linsys_masks = batch['linsys_masks'] # shape: (bsize, Nmaxats, Nmaxats)
        A = e_plus_J + linsys_masks          # shape: (bsize, Nmaxats, Nmaxats)
        
        # b-vector full of zeros
        b_chi = jnp.zeros((bsize, N_linsys)) # shape: (bsize, Nmaxats)
        # Place electronegativities to the RHS
        b_chi = b_chi.at[compind_vec].set(-chi - phi_ext)
        # If Z \neq 0, we need to compute \sum_j Zj / Rij ...
        edotZ = e @ Z[...,None]                    # shape: (bsize, Nmaxats, 1)
        # ... and add it to the RHS
        b = b_chi - edotZ[...,0]
        
        if self.pbc:
            # constraint \sum_i Q_i = \sum_i q_i + Z_i = 0
            b = b.at[:,-1].set(-Z.sum(axis=1)) # 
        else: # finite system, can be non neutral
            tot_qcharge = jax.lax.scatter_add(jnp.zeros(bsize),jnp.expand_dims(batch['inde'][0:real_nats_tot],-1),batch['charges'],dimnums,mode='drop') # shape: (bsize,)
            b = b.at[:,-1].set(tot_qcharge - Z.sum(axis=1)) 
                    
        # solving the linear system
        q = jnp.linalg.solve(A,b[:,:,jnp.newaxis]) # shape: (bsize, Natsmax, 1)
        q = q[...,0]                               # shape: (bsize, Natsmax)
        # total charge = q + Z
        Q = q + Z                                  # shape: (bsize, Natsmax)

        # Removing Lagrange multiplier from q
        qcharges = jnp.zeros_like(q) # shape: (bsize, Natsmax)
        qcharges = qcharges.at[compind_vec].set(q.at[compind_vec].get()) # shape: (bsize, Natsmax)
        q = qcharges.copy() # shape: (bsize, Natsmax)
        # doing the same for Q
        Qcharges = jnp.zeros_like(Q) # shape: (bsize, Natsmax)
        Qcharges = Qcharges.at[compind_vec].set(Q.at[compind_vec].get()) # shape: (bsize, Natsmax)
        Q = Qcharges.copy() # shape: (bsize, Natsmax)
        
        # Flattening charges array and padding it 
        ind = jnp.where(qcharges != 0, fill_value=-1, size=nats) # indices, shape: (Natspad, Natspad)
        qcharges = qcharges[ind]                    # shape: (Natspad,)
        Zcharges = Z[ind]                           # shape: (Natspad,)
        # total charge = q + Z
        Qcharges = qcharges + Zcharges              # shape: (Natspad,)
        Qcharges = Qcharges[0:real_nats_tot]        # shape: (real_nats_tot,)
        # ---------------------


        # --- Energy calculation ---
        # El energy calculation for the batch
        Elr = 1/2 * jnp.einsum('bi,bij,bj->b', Q, e, Q) # shape: (bsize,)
        # self term w/ hardness: \sum_i 1/2  Ji qi^2  
        E2 = jax.lax.scatter_add(jnp.zeros(bsize), jnp.expand_dims(batch['inde'][0:real_nats_tot],-1), 
                                 1/2 * J * qcharges[0:real_nats_tot]**2, dimnums,mode='drop') # shape: (bsize,)
        # In case the system is not neutral, remove Qtot from b
        b = b.at[:,-1].set(0.0) 
        # self term w/ electronegativity: \sum_i chi_i qi (+ \phi^{ext}_i qi if there is)
        E1 = jnp.sum(-b_chi*q, axis=1) # shape: (bsize,)
        # external term \sum_i Z_i \phi^{ext}_i
        EZext = jax.lax.scatter_add(jnp.zeros(bsize), jnp.expand_dims(inde, -1), phi_ext * Zref, dimnums, mode='drop') 
        # summing sr and lr energy (J is already contained in the diagonals of Elr)
        E = E0 + E1 + E2 + Elr + EZext         # shape: (bsize,)
        # --------------------------
        
        
        if not self.forces: # training only on energies
            return [E, 0, 0, Qcharges] 
        
        # ---- Forces ----
        # purely short range forces
        F0 = gather_forces(operand, inda, indb, Fterms0) # shape: (Natspad, 3)
        # terms O(q): \sum_i \chi_i q_i 
        # F1 = gather_forces(operand, inda, indb, Fterms1) * qcharges[:,None] # shape: (Natspad, 3)
        (weighted_Fterms1,) = Ei1_pullback(jax.lax.stop_gradient(qcharges))# jnp.ones_like(Ei1)
        # jax.debug.print('wwww {} ', weighted_Fterms1.shape)
        # jax.debug.print('Ei1 {} ', Ei1.shape)
        # jax.debug.print('qqqq {} ', qcharges.shape)
        F1 = gather_forces(operand, inda, indb, weighted_Fterms1)  # shape: (Natspad, 3)
        # jax.debug.print('ffff {} ', F1.shape)
        
        # terms O(q^2)
        # F2 = gather_forces(operand, inda, indb, Fterms2) * 1/2 * qcharges[:,None]**2 # shape: (Natspad, 3)
        (weighted_Fterms2,) = Ei2_pullback(jax.lax.stop_gradient(1/2 * qcharges**2)) # jnp.ones_like(Ei1)
        F2 = gather_forces(operand, inda, indb, weighted_Fterms2)  # shape: (Natspad, 3)
        
        # long range forces
        if self.gaussian_charges:
            out_ew = ew_fs(coords,
                            latvecs,
                            gauss_radii,
                            alphas_screen,
                            rcuts,
                            Gcuts,
                            q=Q)
        else:
            out_ew = ew_fs(coords,
                            latvecs,
                            alphas_screen,
                            rcuts,
                            Gcuts,
                            q=Q)
            
        # long range forces
        # Flr = jnp.einsum('bi, bijx, bj -> bix', Q,f,Q)        # shape: (bsize, Natsmax, 3)
        Flr = out_ew['forces']
            
        ind = jnp.where(Flr != 0, fill_value=-1, size=nats*3) # shape: (3, Natspad*3)
        Flr = Flr[ind].reshape((nats,3))                      # shape: (Natspad, 3)
        F = F0 + F1 + F2 + Flr # shape: (Natspad, 3)
        
        if is_Efield_applied: # then add forces from external electric field, qi Eαi
            Fext = Qcharges[:,None] * Efield_ext 
            F += Fext
        # ----------------

        # # jax.debug.print('new aaa charges MgO: {}  O: {}', Qcharges[0], Qcharges[-1])
        extra['J'] = J
        # extra['kernel'] = e[:,0:-1,0:-1]
        # extra['mu'] = q[:,-1]
        # extra['A'] = A
        # extra['b'] = b
        extra['e'] = e
        extra['Fterms0'] = Fterms0
        extra['Ei0'] = Ei0
        
        
        return E, F, [0], Qcharges, 0, extra
    


def E_func(x, sp, layers):
    for i,l in enumerate(layers):
        x = l(sp,x)
    return x

def pairs2E(disp, batch, dfunc, layers, i=0):
    des = dfunc(disp, batch)
    Ei = jax.vmap(E_func, in_axes=(0,0,None))(des,batch['species'],layers)
    Ei = Ei[:,i] 
    return jnp.sum(Ei), Ei

def pairs2Ei(disp, batch, dfunc, layers, i=0):
    des = dfunc(disp, batch)
    Ei = jax.vmap(E_func, in_axes=(0,0,None))(des,batch['species'],layers)
    Ei = Ei[:,i] 
    return Ei
