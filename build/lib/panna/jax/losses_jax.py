import jax.numpy as jnp

# Written with E and F square loss in mind for now
def compute_loss(parameters, modapp, model_variables, model_fixed, batch):
    model_state = {**model_variables,**model_fixed}
    preds = modapp(model_state, batch)
    if parameters['num_targets']==1:
        Ecost = parameters['energy_cost']
        Fcost = parameters['forces_cost']
    else:
        Ecost = parameters['energy_cost'][batch['conf_targets']]
        Fcost = jnp.expand_dims(parameters['forces_cost'][batch['targets']],-1)
    if parameters['energy_loss'] == 'per_mol':
        Eloss = jnp.mean(Ecost*(preds[0]-batch['energy'])**2)
    elif parameters['energy_loss'] == 'per_at':
        Eloss = jnp.mean(Ecost*((preds[0]-batch['energy'])/batch['nats'].astype(jnp.float32))**2)
    if parameters['forces']:
        # Just using the mean per component
        Floss = jnp.sum(Fcost*(preds[1]-batch['forces'])**2)/(3*batch['ntot']).astype(jnp.float32)
    fom = {}
    if 'MAE' in parameters['metrics']:
        fom['MAEat'] = jnp.mean(jnp.abs(preds[0]-batch['energy'])/batch['nats'].astype(jnp.float32))
        if parameters['forces']:
            fom['FMAEcomp'] = jnp.sum(jnp.abs(preds[1]-batch['forces']))/\
                              jnp.sum((3*batch['nats']),dtype=jnp.float32)
    if 'RMSE' in parameters['metrics']:
        fom['RMSEat'] = jnp.sqrt(jnp.mean(((preds[0]-batch['energy'])/batch['nats'].astype(jnp.float32))**2))
        if parameters['forces']:
            fom['FRMSEcomp'] = jnp.sqrt(jnp.sum((preds[1]-batch['forces'])**2)/\
                              jnp.sum((3*batch['nats']),dtype=jnp.float32))
    if parameters['forces']:
        return Eloss + Floss, fom
    else:
        return Eloss, fom

def compute_diffsums(model_state, modapp, parameters, batch):
    preds = modapp(model_state, batch)
    fom = {}
    if 'MAE' in parameters['metrics']:
        fom['EAsumat'] = jnp.sum(jnp.abs(preds[0]-batch['energy'])/batch['nats'].astype(jnp.float32))
        if parameters['forces']:
            fom['FAsum'] = jnp.sum(jnp.abs(preds[1]-batch['forces']))
    if 'RMSE' in parameters['metrics']:
        fom['E2sumat'] = jnp.sum(((preds[0]-batch['energy'])/batch['nats'].astype(jnp.float32))**2)
        if parameters['forces']:
            fom['F2sum'] = jnp.sum((preds[1]-batch['forces'])**2)            
    fom['Nasum'] = jnp.sum(batch['nats']).astype(jnp.float32)
    fom['bs'] = batch['nats'].shape[0]
    return fom