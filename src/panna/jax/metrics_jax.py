import numpy as np
from collections import OrderedDict

class metrics():
    def __init__(self, parameters):
        self.mix = parameters['mix_factor']
        self.omix = 1.0-self.mix
        if parameters['val_data_dir']:
            self.val = True
        else:
            self.val = False
        self.forces = parameters['forces']
        self.MAE = 'MAE' in parameters['metrics']
        self.RMSE = 'RMSE' in parameters['metrics']

        # Dict with one entry for each fom in training containing:
        # [average, output_name, corresponding_loss_fom_name]
        # If we add a new option it's probably enough to change here.
        self.train_foms = OrderedDict()
        if self.MAE:
            self.train_foms['MAE'] = [0.0, 'MAE/at', 'MAEat']
            if self.forces:
                self.train_foms['FMAE'] = [0.0, 'MAEF', 'FMAEcomp']
        if self.RMSE:
            self.train_foms['RMSE'] = [0.0, 'RMSE/at', 'RMSEat']
            if self.forces:
                self.train_foms['FRMSE'] = [0.0, 'RMSEF', 'FRMSEcomp']

        # Dict with one entry for each fom in validation
        # since we need to average after, this has the form:
        # [average_numerator, average_denominator, output_name, 
        # numerator_diffsum_fom_name, denominator_diffsum_fom_name,
        # function to apply]
        self.val_foms = OrderedDict()        
        if self.MAE:
            self.val_foms['MAE'] = [0.0, 0.0, 'Val_MAE/at', 'EAsumat', 'bs', 
                                    lambda x,y:x/y]
            if self.forces:
                self.val_foms['FMAE'] = [0.0, 0.0, 'Val_MAEF', 'FAsum', 'Nasum',
                                         lambda x,y:x/(3*y)]
        if self.RMSE:
            self.val_foms['RMSE'] = [0.0, 0.0, 'Val_RMSE/at', 'E2sumat', 'bs', 
                                    lambda x,y:np.sqrt(x/y)]
            if self.forces:
                self.val_foms['FRMSE'] = [0.0, 0.0, 'Val_RMSEF', 'F2sum', 'Nasum',
                                          lambda x,y:np.sqrt(x/(3*y))]


    # The header for the metrics file
    def header(self):
        header = '#epoch\tstep\t'
        header += '\t'.join([f[1] for f in self.train_foms.values()])
        if self.val:
            header += '\t'+'\t'.join([f[2] for f in self.val_foms.values()])
        header += '\n'
        return header

    # Updating the training averages
    # (no averaging in the first step)
    def update_train(self, fom, first=False):
        for f in self.train_foms.values():
            if first:
                f[0] = fom[f[2]]
            else:
                f[0] = self.omix*f[0] + self.mix*fom[f[2]]

    # Output string for command line
    def train_string(self):
        return '  '.join([f[1]+': '+str(f[0]) for f in self.train_foms.values()])

    # Output train foms    
    def train_list(self):
        return [f[0] for f in self.train_foms.values()]

    # Updating the validation averages
    # (no averaging in the first step)
    def update_val(self, fom):
        for f in self.val_foms.values():
            f[0] += fom[f[3]]
            f[1] += fom[f[4]]

    # Computing final averages,
    # creating string and list output,
    # then resetting for next batch
    def finalize_val(self):
        out_str = ''
        out_list = []     
        for f in self.val_foms.values():
            out = f[5](f[0],f[1])
            out_str += f[2]+': '+str(out)+' '
            out_list.append(out)
            # Reset
            f[0] = 0.0
            f[1] = 0.0
        return out_str, out_list




