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
        self.forces = parameters['forces_cost']>0.0
        self.stress = parameters['stress_cost']>0.0
        self.charges = parameters['charges_cost']>0.0
        self.dipoles = parameters['dipoles_cost']>0.0
        self.MAE = 'MAE' in parameters['metrics']
        self.RMSE = 'RMSE' in parameters['metrics']
        self.percent_RMSE = 'percent_RMSE' in parameters['metrics']
        self.dump_avg_charges = 'avg_charges' in parameters['metrics']
        self.dump_extra_stuff = [key for key in parameters['metrics'] if key not in ['MAE', 'RMSE', 'percent_RMSE', 'avg_charges', 'loss']] 
        self.loss = 'loss' in parameters['metrics']
        
        # Dict with one entry for each fom in training containing:
        # [average, output_name, corresponding_loss_fom_name]
        # If we add a new option it's probably enough to change here.
        self.train_foms = OrderedDict()
        if self.MAE:
            self.train_foms['MAE'] = [0.0, 'MAE/at', 'MAEat']
            if self.forces:
                self.train_foms['FMAE'] = [0.0, 'MAEF', 'FMAEcomp']
            if self.stress:
                self.train_foms['SMAE'] = [0.0, 'MAES', 'SMAEcomp']
            if self.charges:
                self.train_foms['QMAE'] = [0.0, 'MAEQ', 'QMAEat']
            if self.dipoles:
                self.train_foms['DMAE'] = [0.0, 'MAED', 'DMAEcomp']
        if self.RMSE:
            self.train_foms['RMSE'] = [0.0, 'RMSE/at', 'RMSEat']
            if self.forces:
                self.train_foms['FRMSE'] = [0.0, 'RMSEF', 'FRMSEcomp']
            if self.stress:
                self.train_foms['SRMSE'] = [0.0, 'RMSES', 'SRMSEcomp']
            if self.charges:
                self.train_foms['QRMSE'] = [0.0, 'RMSEQ', 'QRMSEat']
            if self.dipoles:
                self.train_foms['DRMSE'] = [0.0, 'RMSED', 'DRMSEcomp']
        if self.percent_RMSE:
            self.train_foms['%RMSE'] = [0.0, 'relRMSE/at', '%RMSEat']
            if self.forces:
                self.train_foms['%FRMSE'] = [0.0, 'relRMSEF', '%FRMSEcomp']
            if self.stress:
                self.train_foms['%SRMSE'] = [0.0, 'relRMSES', '%SRMSEcomp']
            if self.charges:
                self.train_foms['%QRMSE'] = [0.0, 'relRMSEQ', '%QRMSEat']
            if self.dipoles:
                self.train_foms['%DRMSE'] = [0.0, 'relRMSED', '%DRMSEcomp']
        if self.dump_avg_charges:
            for sp in parameters['species']:
                self.train_foms[f'avg charge {sp}'] = [0.0, f'avg_charge_{sp}', f'avg_charge_{sp}']
        if len(self.dump_extra_stuff) > 0:
            for key in self.dump_extra_stuff:
                for sp in parameters['species']:
                    self.train_foms[f'{key} {sp}'] = [0.0, f'{key}_{sp}', f'{key}_{sp}']
        if self.loss:
            self.train_foms['Floss'] = [0.0, 'Floss', 'Floss']
            self.train_foms['Bloss'] = [0.0, 'Bloss', 'Bloss']
            self.train_foms['Sloss'] = [0.0, 'Sloss', 'Sloss']

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
            if self.stress:
                self.val_foms['SMAE'] = [0.0, 0.0, 'Val_MAES', 'Ssum', 'bs',
                                         lambda x,y:x/(9*y)]
                
            if self.charges:
                self.val_foms['QMAE'] = [0.0, 0.0, 'Val_MAEQ', 'QAsum', 'Nasum',
                                         lambda x,y:x/y]
            if self.dipoles:
                self.val_foms['DMAE'] = [0.0, 0.0, 'Val_MAED', 'DAsum', 'Nasum',
                                         lambda x,y:x/(3*y)]
                
        if self.RMSE:
            self.val_foms['RMSE'] = [0.0, 0.0, 'Val_RMSE/at', 'E2sumat', 'bs', 
                                    lambda x,y:np.sqrt(x/y)]
            if self.forces:
                self.val_foms['FRMSE'] = [0.0, 0.0, 'Val_RMSEF', 'F2sum', 'Nasum',
                                          lambda x,y:np.sqrt(x/(3*y))]
            if self.stress:
                self.val_foms['SRMSE'] = [0.0, 0.0, 'Val_RMSES', 'S2sum', 'bs',
                                          lambda x,y:np.sqrt(x/(9*y))]
            if self.charges:
                self.val_foms['QRMSE'] = [0.0, 0.0, 'Val_RMSEQ', 'Q2sum', 'Nasum',
                                         lambda x,y:np.sqrt(x/y)]
            if self.dipoles:
                self.val_foms['DRMSE'] = [0.0, 0.0, 'Val_RMSED', 'D2sum', 'Nasum',
                                          lambda x,y:np.sqrt(x/(3*y))]
        if self.percent_RMSE:
            self.val_foms['%RMSE'] = [0.0, 0.0, 'Val_relRMSE/at', '%E2sumat', 'bs', 
                                    lambda x,y:np.sqrt(x/y)]
            if self.forces:
                self.val_foms['%FRMSE'] = [0.0, 0.0, 'Val_relRMSEF', '%F2sum', 'F2sumtot',
                                          lambda x,y:np.sqrt(x/y)]
            if self.stress:
                self.val_foms['%SRMSE'] = [0.0, 0.0, 'Val_relRMSES', '%S2sum', 'S2sumtot',
                                          lambda x,y:np.sqrt(x/y)]
            if self.charges:
                self.val_foms['%QRMSE'] = [0.0, 0.0, 'Val_relRMSEQ', '%Q2sum', 'Q2sumtot',
                                         lambda x,y:np.sqrt(x/y)]
            if self.dipoles:
                self.val_foms['%DRMSE'] = [0.0, 0.0, 'Val_relRMSED', '%D2sum', 'D2sumtot',
                                          lambda x,y:np.sqrt(x/y)]
        if self.dump_avg_charges:
            for sp in parameters['species']:
                self.val_foms[f'avg charge {sp}'] = [0.0, 0.0, f'Val_avg_charge_{sp}', f'sum_charge_{sp}', f'nats_{sp}', lambda x,y:x/y]
        
        if len(self.dump_extra_stuff) > 0:
            for key in self.dump_extra_stuff:
                for sp in parameters['species']:
                    self.val_foms[f'{key} {sp}'] = [0.0, 0.0, f'Val_{key}_{sp}', f'sum_{key}_{sp}', f'nats_{sp}', lambda x,y:x/y]
            


    # The header for the metrics file
    def header(self):
        header = '#epoch\tstep\t'
        header += '\t'.join([f[1] for f in self.train_foms.values()])
        if self.val:
            header += '\t'+'\t'.join([f[2] for f in self.val_foms.values()])
        # prefix = ['']
        # if self.val:
        #     prefix.append('Val_')
        # for pre in prefix:
        #     for f in self.train_foms.values():
        #         header += pre+f[1]+'\t'
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




# class metrics_LR():
#     def __init__(self, parameters):
#         self.mix = parameters['mix_factor']
#         self.omix = 1.0-self.mix
#         if parameters['val_data_dir']:
#             self.val = True
#         else:
#             self.val = False
#         self.forces = parameters['forces_cost']>0.0
#         self.MAE = 'MAE' in parameters['metrics']
#         self.RMSE = 'RMSE' in parameters['metrics']

#         # Dict with one entry for each fom in training containing:
#         # [average, output_name, corresponding_loss_fom_name]
#         # If we add a new option it's probably enough to change here.
#         self.train_foms = OrderedDict()
#         if self.MAE:
#             self.train_foms['MAE'] = [0.0, 'MAE/at', 'MAEat']
#             if self.forces:
#                 self.train_foms['FMAE'] = [0.0, 'MAEF', 'FMAEcomp']
#             self.train_foms['QMAE'] = [0.0, 'MAEQ', 'QMAE']
#         if self.RMSE:
#             self.train_foms['RMSE'] = [0.0, 'RMSE/at', 'RMSEat']
#             if self.forces:
#                 self.train_foms['FRMSE'] = [0.0, 'RMSEF', 'FRMSEcomp']
#             self.train_foms['QRMSE'] = [0.0, 'RMSEQ', 'QRMSE']

#         # Dict with one entry for each fom in validation
#         # since we need to average after, this has the form:
#         # [average_numerator, average_denominator, output_name, 
#         # numerator_diffsum_fom_name, denominator_diffsum_fom_name,
#         # function to apply]
#         self.val_foms = OrderedDict()        
#         if self.MAE:
#             self.val_foms['MAE'] = [0.0, 0.0, 'Val_MAE/at', 'EAsumat', 'bs', 
#                                     lambda x,y:x/y]
#             if self.forces:
#                 self.val_foms['FMAE'] = [0.0, 0.0, 'Val_MAEF', 'FAsum', 'Nasum',
#                                          lambda x,y:x/(3*y)]
#         if self.RMSE:
#             self.val_foms['RMSE'] = [0.0, 0.0, 'Val_RMSE/at', 'E2sumat', 'bs', 
#                                     lambda x,y:np.sqrt(x/y)]
#             if self.forces:
#                 self.val_foms['FRMSE'] = [0.0, 0.0, 'Val_RMSEF', 'F2sum', 'Nasum',
#                                           lambda x,y:np.sqrt(x/(3*y))]


#     # The header for the metrics file
#     def header(self):
#         header = '#epoch\tstep\t'
#         header += '\t'.join([f[1] for f in self.train_foms.values()])
#         if self.val:
#             header += '\t'+'\t'.join([f[2] for f in self.val_foms.values()])
#         # prefix = ['']
#         # if self.val:
#         #     prefix.append('Val_')
#         # for pre in prefix:
#         #     for f in self.train_foms.values():
#         #         header += pre+f[1]+'\t'
#         header += '\n'
#         return header

#     # Updating the training averages
#     # (no averaging in the first step)
#     def update_train(self, fom, first=False):
#         for f in self.train_foms.values():
#             if first:
#                 f[0] = fom[f[2]]
#             else:
#                 f[0] = self.omix*f[0] + self.mix*fom[f[2]]

#     # Output string for command line
#     def train_string(self):
#         return '  '.join([f[1]+': '+str(f[0]) for f in self.train_foms.values()])

#     # Output train foms    
#     def train_list(self):
#         return [f[0] for f in self.train_foms.values()]

#     # Updating the validation averages
#     # (no averaging in the first step)
#     def update_val(self, fom):
#         for f in self.val_foms.values():
#             f[0] += fom[f[3]]
#             f[1] += fom[f[4]]

#     # Computing final averages,
#     # creating string and list output,
#     # then resetting for next batch
#     def finalize_val(self):
#         out_str = ''
#         out_list = []     
#         for f in self.val_foms.values():
#             out = f[5](f[0],f[1])
#             out_str += f[2]+': '+str(out)+' '
#             out_list.append(out)
#             # Reset
#             f[0] = 0.0
#             f[1] = 0.0
#         return out_str, out_list




