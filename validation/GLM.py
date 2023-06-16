import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import Ridge
from os.path import join as pjoin
from nilearn.glm.first_level import make_first_level_design_matrix
from had_utils import save2cifti

# define path
# make sure the dataset_path are modified based on your personal dataset downloading directory
dataset_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/data_upload/HAD'
ciftify_path = f'{dataset_path}/derivatives/ciftify'
nifti_path = f'{dataset_path}'
support_path = './support_files'
result_path = './results'
save_path = pjoin(result_path, 'beta')
if not os.path.exists(save_path):
    os.makedirs(save_path)
# change to path of current file
os.chdir(os.path.dirname(__file__))

# prepare params
sub_names = ['sub-%02d'%(i+1) for i in range(30)]
# sub_names = ['sub-24']
tr, begin_dur, n_tr, n_event, n_run, n_cycle, n_class = 2, 12, 156, 60, 12, 4, 180
frame_times = np.arange(n_tr*3) * tr 
sess_name = 'ses-action01'
alpha = 0.1

# start modeling
for sub_idx, sub_name in enumerate(sub_names):
    # prepare basic path
    sub_nifti_path = pjoin(nifti_path, sub_name)
    result_path = pjoin(ciftify_path, sub_name, 'results')
    sub_func_path = pjoin(sub_nifti_path, sess_name, 'func')
    events_file = sorted([i for i in os.listdir(sub_func_path) if i.endswith('.csv')])
    # loop in cycles to perform linear regression model
    for cycle_idx in range(n_cycle):
        beta = np.zeros((n_class, 59412), dtype=np.float32)
        # intial containers
        trial_type_cycle = []
        stim_names_cycle = []
        dtseries_cycle = np.zeros((3*n_tr, 91282))
        onset_cycle = np.zeros((3*n_event))
        duration_cycle = np.zeros((3*n_event))
        for run_idx in range(3):
            run_file_idx = run_idx + cycle_idx * 3 + 1
            events_file_name = '%s_ses-action01_task-action_run-%02d_events.tsv'%(sub_name, run_file_idx)
            run_name = 'ses-action01_task-action_run-%d'% run_file_idx
            # fit design matrix based on trial onset time
            events_raw = pd.read_csv(pjoin(sub_func_path, events_file_name), sep='\t')
            duration = events_raw['duration']
            onset = events_raw['onset'].to_numpy() + begin_dur
            label_tmp = events_raw['trial_type'].to_numpy()
            trial_type = ['category-%03d'%idx for idx in label_tmp]
            stim_names = [_.split('/')[-1] for _ in events_raw['stim_file']]
            # load time series
            dtseries_path = pjoin(result_path, f'{run_name}_Atlas.dtseries.nii')
            dtseries = nib.load(dtseries_path).get_fdata()
            # concantenate all info into cycle params
            dtseries_cycle[n_tr*run_idx:n_tr*(run_idx+1)] = dtseries
            onset_cycle[n_event*run_idx:n_event*(run_idx+1)] = onset + run_idx * n_tr * tr
            duration_cycle[n_event*run_idx:n_event*(run_idx+1)] = duration
            trial_type_cycle.extend(trial_type)
            stim_names_cycle.extend(stim_names)
        # prepare design matrix
        events = pd.DataFrame({'trial_type':trial_type_cycle, 'onset':onset_cycle, 'duration':duration_cycle})
        design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
        design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
        # add drift columns
        drift_order = 2
        frame_times_single = np.arange(n_tr) * tr 
        drift_effect = np.zeros((n_tr * 3, 3*(drift_order+1)))
        tmax = float(frame_times_single.max())
        for run_idx in range(3):
            for k in range(drift_order+1):
                drift_effect[n_tr*run_idx:n_tr*(run_idx+1), (drift_order+1)*run_idx+k] = (frame_times_single / tmax) ** k
        drift_effect = pd.DataFrame(drift_effect)
        # concantenate 
        design_matrix = pd.concat([design_matrix.reset_index(drop=True), drift_effect], ignore_index=True, axis=1)  
        # perform GLM
        reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries_cycle[:, :59412])
        beta = reg.coef_[:, :n_class].transpose(1,0).astype(np.float32)
        print('Finish performing GLM in %s %s model %02d in alpha %.3f'%(sub_name, sess_name, cycle_idx+1, alpha))
        # save beta and label information
        beta_path = pjoin(result_path, f'ses-action01_task-action_cycle-{cycle_idx+1}_beta.dscalar.nii')
        label_path = pjoin(result_path, f'ses-action01_task-action_cycle-{cycle_idx+1}_label.txt')
        # sort stim names info and write into txt 
        stim_names_cycle.sort(key=lambda x:x.split('_')[1])
        with open(label_path, 'w') as f:
            f.writelines([line+'\n' for line in stim_names_cycle])
        # save beta
        beta_path = pjoin(save_path, f'{sub_name}_cycle-{cycle_idx+1}_beta.dscalar.nii')
        temp = nib.load(pjoin(support_path, 'template.dtseries.nii'))
        bm = list(temp.header.get_index_map(1).brain_models)[0:2]
        save2cifti(file_path=beta_path, data=beta, brain_models=bm, map_names=stim_names_cycle)