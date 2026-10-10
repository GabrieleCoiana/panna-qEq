import jax.numpy as jnp
import jax

Ang_to_bohr = 1.88973
ev_to_Ha = 0.03674929


def MSE(y,y0):
    return jnp.mean( (y-y0)**2 )

def MAE(y,y0):
    return jnp.mean( jnp.abs(y-y0) )

def RMSE(y,y0, size):
    return jnp.sqrt( 1/size*( (y-y0)**2 ) )

def percent_RMSE(y,y0):
    num = jnp.sum( (y-y0)**2 )
    den = jnp.sum( y0**2 )
    # rel_diff2 = jnp.where(y != 0, (y-y0)**2 / y0**2, 0)   # (Npad, 3)
    # rel_diff = jnp.sqrt( jnp.sum(rel_diff2, axis=1) )     # (Npad)
    return jnp.sqrt(num / den)#1/size * jnp.sum(rel_diff)


# Written with E and F square loss in mind for now
def compute_loss(parameters, modapp, model_variables, model_fixed, batch):
    model_state = {**model_variables,**model_fixed}
    preds = modapp(model_state, batch)
    nats = batch['nats'].astype(jnp.float32) if parameters['BORN'] == '' else batch['nats'][0:-1].astype(jnp.float32) 
    natstot = jnp.sum(nats)

    nats_BORN = 0
    nspecies = len(parameters['species'])
    species = batch['species']
    bsize = jnp.shape(batch['nats'])[0]
    
    Floss, Sloss, Qloss, Dloss, Bloss = 0, 0, 0, 0, 0
    
    if parameters['energy_loss'] == 'per_mol':
        Eloss = jnp.mean((preds[0]-batch['energy'])**2)
    elif parameters['energy_loss'] == 'per_at':
        print('LOSS - Energy loss')
        Eloss = jnp.mean( ((preds[0]-batch['energy'])/nats)**2 ) 
    
    if parameters['forces_cost']>0.0:
        print('LOSS - Forces loss')
        # Just using the mean
        Floss = jnp.sum((preds[1]-batch['forces'])**2)/ (3*natstot)
    
    if parameters['stress_cost'] > 0.0:
        print('LOSS - Stress loss')
        Sloss = 1/(9*bsize) * jnp.sum((preds[2] - batch['stress'])**2)
    
    # added for LR
    if parameters['charges_cost']>0.0:
        Qloss = jnp.sum((preds[3]-batch['charges'])**2) / natstot
    
    # added for LR pol
    if parameters['dipoles_cost']>0.0:
        Dloss = jnp.sum((preds[4]-batch['dipoles'])**2) / (3*natstot)
    
    # added for training on BEC
    if parameters['BORN'] != '':
        print('LOSS - Born loss')
        
        nats_BORN = int(batch['born_charges'].size // 9)
        real_nats_tot = jnp.shape(batch['indf'])[0] if parameters['BORN'] == '' else jnp.shape(batch['indf'])[0] - nats_BORN
        Omega = jnp.linalg.det(batch['latvecs'][-1]) * Ang_to_bohr**3
        
        # getting result from the unperturbed configuration
        batch['Efield_ext'] = jnp.zeros((nats_BORN, 3)) 
        _, _, _, _,_, extra  = modapp(model_state, batch)
        dipoles0 = extra['d_BORN'] 
        forces0 = extra['force_BORN'] 
        # jax.debug.print('dipoles 0 {} ', dipoles0)
        # jax.debug.print('\nforces 0 {} ', forces0)
        
        Alpha = jnp.zeros((3, 3))
        born_charges = jnp.zeros((nats_BORN,3,3))
        for beta in range(3): # apply Efield in the 3 directions
            Efield = jnp.zeros((nats_BORN, 3))
            Efield = Efield.at[:,beta].set(0.001) # small value --> linear response
            batch['Efield_ext'] = Efield
            
            # getting result from perturbed configuration
            _, _, _, _,_ ,extrad = modapp(model_state, batch)
            dipolesd = extrad['d_BORN'] 
            forcesd = extrad['force_BORN'] 
            # Efield_int = np.array(preds[4]['Efield'])  # (nats, 3)
        
            # Pol = \sum_i d_i
            Alpha = Alpha.at[:,beta].set(jnp.sum(dipolesd - dipoles0, axis=0) * Ang_to_bohr / (0.001 * ev_to_Ha / Ang_to_bohr)) # shape: (3,)
            
            born = (forcesd - forces0) / 0.001
            born_charges = born_charges.at[:,:,beta].set(born)
            
        eps = jnp.eye(3) + 4*jnp.pi/Omega * Alpha
        # jax.debug.print('\neeeeps {}', eps)
        # jax.debug.print('\n boooorn {}', born_charges)
        
        
        # born_ref = batch['born_charges'].reshape((-1,3,3))
        # Zdiag = jnp.diagonal((born_ref), axis1=2)
        # Zmin = jnp.min(Zdiag, axis=1)
        # Zi_trys = 0.4 * Zmin
        # # Zi_trys does not sum up to zero
        # total_charge = jnp.sum(Zi_trys)  # 1*Z_Ba + 3*Z_O + 1*Z_Ti != 0
        # Zi_decided = Zi_trys - 1/nats_BORN * total_charge # rounded to 1st decimal
        # # print('Zi decided ', Zi_decided)
        # Zi_all = Zi_decided[batch['species']][0:real_nats_tot]

        
        # jax.debug.print('charges now {} ', preds[2][0:40:8])
        # jax.debug.print('target      {} ', Zmin_all[0:40:8]*0.6)
        # Bloss = 1/9 * jnp.sum( (eps.ravel() - batch['eps_infty'])**2 ) \
        #         + 1/(9*nats_BORN) * jnp.sum( (born_charges.ravel() - batch['born_charges'])**2 ) #\
        #             # + jnp.sum( (preds[2][0:real_nats_tot] - Zi_all)**2  )
        Bloss =  jnp.sum( (eps.ravel() - batch['eps_infty'])**2 ) \
                 + jnp.sum( (born_charges.ravel() - batch['born_charges'])**2 ) #\
                     # + jnp.sum( (preds[2][0:real_nats_tot] - Zi_all)**2  )
         
        # jax.debug.print('Bloss {}', Bloss)
    
    
    fom = {}
    if 'MAE' in parameters['metrics']:
        fom['MAEat'] = jnp.mean(jnp.abs(preds[0]-batch['energy'])/nats)
        if parameters['forces_cost']>0.0:
            fom['FMAEcomp'] = jnp.sum(jnp.abs(preds[1]-batch['forces']))/\
                              jnp.sum((3*batch['nats']),dtype=jnp.float32)
        if parameters['stress_cost']>0.0:
            fom['SMAEcomp'] = jnp.sum(jnp.abs(preds[2]-batch['stress']))/\
                              (9*bsize) 
        if parameters['charges_cost']>0.0:
            fom['QMAEat'] = jnp.sum(jnp.abs(preds[3]-batch['charges']))/ natstot
        if parameters['dipoles_cost']>0.0:
            fom['DMAEcomp'] = jnp.sum(jnp.abs(preds[4]-batch['dipoles']))/\
                              jnp.sum((3*batch['nats']),dtype=jnp.float32)
            
    if 'RMSE' in parameters['metrics']:
        fom['RMSEat'] = jnp.sqrt(jnp.mean(((preds[0]-batch['energy'])/nats)**2))
        if parameters['forces_cost']>0.0:
            fom['FRMSEcomp'] = jnp.sqrt(jnp.sum((preds[1]-batch['forces'])**2)/\
                              jnp.sum((3*batch['nats']),dtype=jnp.float32))
        if parameters['stress_cost']>0.0:
            fom['SRMSEcomp'] = jnp.sqrt(jnp.sum((preds[2]-batch['stress'])**2)/\
                              (bsize*9))
        if parameters['charges_cost']>0.0:
            fom['QRMSEat'] = jnp.sqrt(jnp.sum((preds[3]-batch['charges'])**2)/ (natstot))
        if parameters['dipoles_cost']>0.0:
            fom['DRMSEcomp'] = jnp.sqrt(jnp.sum((preds[4]-batch['dipoles'])**2)/\
                              jnp.sum((3*batch['nats']),dtype=jnp.float32))
                
    if 'percent_RMSE' in parameters['metrics']:
        fom['%RMSEat'] = percent_RMSE(preds[0]/nats, batch['energy']/nats)        
        if parameters['forces_cost']>0.0:
            fom['%FRMSEcomp'] = percent_RMSE(preds[1], batch['forces'])
        if parameters['stress_cost']>0.0:
            fom['%SRMSEcomp'] = percent_RMSE(preds[2], batch['stress'])
        if parameters['charges_cost']>0.0:
            fom['%QRMSEat'] = percent_RMSE(preds[3], batch['charges'])
        if parameters['dipoles_cost']>0.0:
            fom['%DRMSEcomp'] = percent_RMSE(preds[4], batch['dipoles'])
        
    if 'avg_charges' in parameters['metrics']:
        real_nats_tot = jnp.shape(batch['indf'])[0] if parameters['BORN'] == '' else jnp.shape(batch['indf'])[0] - nats_BORN
        sum_charges, sum_nats = get_avg_perspecie_quantity(real_nats_tot, species, nspecies, preds[3])
        for i,sp in enumerate(parameters['species']):
            fom[f'avg_charge_{sp}'] = sum_charges[i] / sum_nats[i]
    
    # if there are other keys in parameters['metrics'], then take preds[5][key] and average them        
    # NB: preds[4] must contain the key asked!
    extra_metrics = [key for key in parameters['metrics'] if key not in ['MAE', 'RMSE',  'percent_RMSE', 'avg_charges', 'loss']]
    for key in extra_metrics:
        real_nats_tot = jnp.shape(batch['indf'])[0] if parameters['BORN'] == '' else jnp.shape(batch['indf'])[0] - nats_BORN
        sum_key, sum_nats = get_avg_perspecie_quantity(real_nats_tot, species, nspecies, preds[-1][key])
        avg_key = sum_key / sum_nats
        for i,sp in enumerate(parameters['species']):
            fom[f'{key}_{sp}'] = avg_key[i]
        
    
    loss = parameters['energy_cost']*Eloss + \
           parameters['forces_cost']*Floss + \
           parameters['charges_cost']*Qloss + \
           parameters['dipoles_cost']*Dloss + \
           parameters['born_cost']*Bloss +\
           parameters['stress_cost'] * Sloss

    if 'loss' in parameters['metrics']:
        fom['Floss'] = parameters['forces_cost']*Floss
        fom['Bloss'] = parameters['born_cost']* Bloss
        fom['Sloss'] = parameters['stress_cost'] * Sloss
        
    return loss, fom
    


def compute_diffsums(model_state, modapp, parameters, batch):
    preds = jax.jit(modapp)(model_state, batch)
    nats = batch['nats'].astype(jnp.float32) if parameters['BORN'] == '' else batch['nats'][0:-1].astype(jnp.float32) 
    bsize = batch['nats'].shape[0] if parameters['BORN'] == '' else batch['nats'].shape[0] - 1
    nspecies = len(parameters['species'])
    species = batch['species']
    
    fom = {}
    if 'MAE' in parameters['metrics']:
        fom['EAsumat'] = jnp.sum(jnp.abs(preds[0]-batch['energy'])/nats)
        if parameters['forces_cost']>0.0:
            fom['FAsum'] = jnp.sum(jnp.abs(preds[1]-batch['forces']))
        if parameters['stress_cost']>0.0:
            fom['Ssum'] = jnp.sum(jnp.abs(preds[2]-batch['stress']))
        if parameters['charges_cost']>0.0:
            fom['QAsum'] = jnp.sum(jnp.abs(preds[2]-batch['charges']))
        if parameters['dipoles_cost']>0.0:
            fom['DAsum'] = jnp.sum(jnp.abs(preds[3]-batch['dipoles']))
    if 'RMSE' in parameters['metrics']:
        fom['E2sumat'] = jnp.sum(((preds[0]-batch['energy'])/nats)**2)
        if parameters['forces_cost']>0.0:
            fom['F2sum'] = jnp.sum((preds[1]-batch['forces'])**2)  
        if parameters['stress_cost']>0.0:
            fom['S2sum'] = jnp.sum((preds[2]-batch['stress'])**2)  
        if parameters['charges_cost']>0.0:
            fom['Q2sum'] = jnp.sum((preds[3]-batch['charges'])**2)     
        if parameters['dipoles_cost']>0.0:
            fom['D2sum'] = jnp.sum((preds[4]-batch['dipoles'])**2)  
    if 'percent_RMSE' in parameters['metrics']:
        fom['%E2sumat'] = jnp.sum( ((preds[0]-batch['energy'])/nats)**2 / (batch['energy']/nats)**2  )
        if parameters['forces_cost']>0.0:
            # F2sum = jnp.sum( jnp.where(preds[1] != 0, 
                                       # (preds[1]-batch['forces'])**2 / batch['forces']**2  ,
                                        # 0)
                            # , axis=1) # (Npad)
            # Fnorms = jnp.sqrt(F2sum) # (Npad)
            
            fom['%F2sum'] = jnp.sum((preds[1]-batch['forces'])**2)
            fom['F2sumtot'] = jnp.sum(batch['forces']**2)
        if parameters['stress_cost']>0.0:
            fom['%S2sum'] = jnp.sum( (preds[2]-batch['stress'])**2)  
            fom['S2sumtot'] = jnp.sum(batch['stress']**2)
        if parameters['charges_cost']>0.0:
            fom['%Q2sum'] = jnp.sum( (preds[3]-batch['charges'])**2 ) 
            fom['Q2sumtot'] = jnp.sum(batch['charges']**2)
        if parameters['dipoles_cost']>0.0:
            fom['%D2sum'] = jnp.sum( (preds[4]-batch['dipoles'])**2) 
            fom['D2sumtot'] = jnp.sum(batch['dipoles']**2)
    fom['Nasum'] = jnp.sum(nats)
    fom['bs'] = bsize 

    if 'avg_charges' in parameters['metrics']:
        real_nats_tot = jnp.shape(batch['indf'])[0] if parameters['BORN'] == '' else jnp.shape(batch['indf'])[0] - int(batch['born_charges'].size // 9)
        sum_charges, sum_nats = get_avg_perspecie_quantity(real_nats_tot, species, nspecies, preds[3])
        for i,sp in enumerate(parameters['species']):
            fom[f'sum_charge_{sp}'] = sum_charges[i]
            fom[f'nats_{sp}'] = sum_nats[i]
            
    extra_metrics = [key for key in parameters['metrics'] if key not in ['MAE', 'RMSE', 'percent_RMSE', 'avg_charges', 'loss']]
    for key in extra_metrics:
        real_nats_tot = jnp.shape(batch['indf'])[0] if parameters['BORN'] == '' else jnp.shape(batch['indf'])[0] - int(batch['born_charges'].size // 9)
        sum_key, sum_nats = get_avg_perspecie_quantity(real_nats_tot, species, nspecies, preds[-1][key])
        for i,sp in enumerate(parameters['species']):
            fom[f'sum_{key}_{sp}'] = sum_key[i]
            fom[f'nats_{sp}'] = sum_nats[i]
    return fom

def get_avg_perspecie_quantity(real_nats_tot, species, nspecies, quantity):
    # real_nats_tot = jnp.shape(batch['indf'])[0]
    
    dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,),scatter_dims_to_operand_dims=(0,))
    # species = batch['species']
    # nspecies = len(parameters['species']) #2 #len(jnp.unique(batch['species'])) -1
    sum_charges = jax.lax.scatter_add(jnp.zeros(nspecies),species[0:real_nats_tot,None],quantity[0:real_nats_tot],
                            dimnums,mode='drop') # shape: (N_species,)
    sum_nats = jax.lax.scatter_add(jnp.zeros(nspecies), species[0:real_nats_tot,None], jnp.ones(species[0:real_nats_tot].size),
                            dimnums,mode='drop') # shape: (N_species,)
    # jax.debug.print('rrrrr {} {}', real_nats_tot, sum_nats)
    # jax.debug.print('{} {} {} aaaa',sum_charges, sum_nats, sum_charges / sum_nats)
    # avg_charges = sum_charges / sum_nats
    return sum_charges, sum_nats

# # Written with E and F square loss in mind for now
# def compute_loss_LR(parameters, modapp, model_variables, model_fixed, batch):
#     model_state = {**model_variables,**model_fixed}
#     preds = modapp(model_state, batch)
#     nats = batch['nats'].astype(jnp.float32)
#     ntot = batch['ntot'].astype(jnp.float32)
    
#     if parameters['energy_loss'] == 'per_mol':
#         Eloss = MSE( preds[0], batch['energy'] ) 
#     elif parameters['energy_loss'] == 'per_at':
#         Eloss = MSE( preds[0]/nats, batch['energy']/nats )
    
#     if parameters['forces_cost']>0.0:
#         # Just using the mean
#         Floss = jnp.sum((preds[1]-batch['forces'])**2)  / (3*ntot)
#     else:
#         Floss = 0.0
    
#     # added for LR
#     if parameters['charges_cost']>0.0:
#         ntot = batch['ntot'].astype(jnp.float32)
#         Qloss = jnp.sum( MSE(preds[2], batch['charges']) ) / ntot
#     else:
#         Qloss = 0.0
    
#     fom = {}
#     if 'MAE' in parameters['metrics']:
#         fom['MAEat'] = MAE(preds[0]/nats , batch['energy']/nats)
#         # if parameters['forces_cost']>0.0:
#         fom['FMAEcomp'] = jnp.sum( jnp.abs(preds[1]-batch['forces']) ) / (3*ntot)
#         fom['QMAE'] = MAE(preds[2], batch['charges'])
    
#     if 'RMSE' in parameters['metrics']:
#         fom['RMSEat'] = RMSE(preds[0]/nats, batch['energy']/nats)
#         # if parameters['forces_cost']>0.0:
#         fom['FRMSEcomp'] = jnp.sqrt( jnp.sum((preds[1]-batch['forces'])**2) /(3*ntot) )
#         fom['QRMSE'] = RMSE(preds[2], batch['charges'])
    
#     return parameters['energy_cost']*Eloss + parameters['forces_cost']*Floss +parameters['charges_cost']*Qloss, fom
    
