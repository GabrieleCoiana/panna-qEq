import os
import pickle
import glob
import numpy as np

def validate(parameters, data, modapp):
    task = parameters['task']
    print('Looking models in {}/models'.format(parameters['train_dir']))
    files = glob.glob(parameters['train_dir']+'/models/epoch*pkl')
    if len(files)>0:
        namest = [(f,int(f.split('step_')[-1].split('.')[0])) \
                  for f in files \
                  if f.split('step_')[-1].split('.')[0].isnumeric()]
        files = [namest[f][0] for f in np.argsort([n[1] for n in namest])]
    else:
        print('Could not find any model.')
        exit()
    if not os.path.exists(parameters['val_dir']):
        os.mkdir(parameters['val_dir'])
    for wf in files:
        outn = wf.split('/')[-1].split('.pkl')[0]
        if task=='val':
            Efname = parameters['val_dir']+'/'+outn+'.dat'
        elif task=='descr':
            Efname = parameters['val_dir']+'/'+outn+'_descr.pkl'
        if os.path.isfile(Efname):
            print('File {} already exists, skipping this model.'.format(Efname))
            continue
        print('Evaluating '+wf)
        with open(wf, 'rb') as f:
            model_state = pickle.load(f)
        if task=='val':
            ef = open(Efname, 'w')
            ef.write('#filename n_atoms e_ref e_nn\n')
            if parameters['forces']:
                ff = open(parameters['val_dir']+'/'+outn+'_forces.dat', 'w')
                ff.write('#filename atom_id fx_nn fy_nn fz_nn fx_ref fy_ref fz_ref\n')
        elif task=='descr':
            descrd = {}
        for d in data:
            batch = {k:d[k].numpy() for k in d.keys() if k!='name'}
            if task=='val':
                preds = modapp(model_state, batch)
                for en,n,nat,er in zip(preds[0],d['name'].numpy(),\
                                       d['nats'].numpy(),d['energy'].numpy()):
                    ef.write('{} {} {:.6f} {:.6f}\n'.format(n.decode('utf-8').replace(" ", "_"),nat,er,en))
                if parameters['forces']:
                    i = 0
                    fref = d['forces'].numpy()
                    for n,nat in zip(d['name'].numpy(),d['nats'].numpy()):
                        for k in range(nat):
                            ff.write(' '.join([str(n) for n in \
                                     [n.decode('utf-8').replace(" ", "_"),k]+list(preds[1][i+k])+
                                                           list(fref[i+k])])+'\n')
                        i += nat
            elif task=='descr':
                dess = modapp(model_state, batch)
                # Create lists of configurations (plus padding) and scatter the descriptors
                nconfs = len(d['name'].numpy())+1
                dlist = [[] for _ in range(nconfs)]
                for des, e in zip(dess, batch['inde']):
                    dlist[e].append(des)
                for name, des in zip(d['name'].numpy(),dlist[:-1]):
                    descrd[name.decode('utf-8').replace(" ", "_")] = np.asarray(des)
        if task=='val':
            ef.close()
            if parameters['forces']:
                ff.close()
        elif task=='descr':
            with open(Efname, 'wb') as f:
                pickle.dump(descrd, f)

