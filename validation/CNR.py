import os
import numpy as np
import pandas as pd
import scipy.io as sio
import nibabel as nib
from sklearn.linear_model import Ridge
from os.path import join as pjoin
from nilearn.glm.first_level import make_first_level_design_matrix
from had_utils import save2cifti, roi_mask

# define path
# make sure the dataset_path are modified based on your personal dataset downloading directory
dataset_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/data_upload/HAD'
ciftify_path = f'{dataset_path}/derivatives/ciftify'
nifti_path = f'{dataset_path}'
support_path = './support_files'
result_path = './results'
save_indiv_path = pjoin(result_path, 'brain_map_individual')
# change to path of current file
os.chdir(os.path.dirname(__file__))

# prepare params
sub_names = ['sub-%02d'%(i+1) for i in range(30)]
# sub_names = ['sub-01']
tr, begin_dur, n_tr, n_event, n_run, n_cycle, n_class = 2, 12, 156, 60, 12, 4, 180
frame_times = np.arange(n_tr*3) * tr
sess_name = 'ses-action01'
n_sub = len(sub_names)
alpha = 0.1
# template for saving dtseries
temp = nib.load(pjoin(support_path, 'template.dtseries.nii'))

# for alpha in alphas:
cnr_path = pjoin(result_path, f'cnr.dtseries.nii')
if not os.path.exists(cnr_path):
    cnr_sum = np.zeros((n_sub, n_cycle, 59412))
    for sub_idx, sub_name in enumerate(sub_names):
        # prepare basic path
        sub_nifti_path = pjoin(nifti_path, sub_name)
        result_path = pjoin(ciftify_path, sub_name, 'results')
        sub_func_path = pjoin(sub_nifti_path, sess_name, 'func')
        events_file = sorted([i for i in os.listdir(
            sub_func_path) if i.endswith('.csv')])
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
                events_file_name = '%s_ses-action01_task-action_run-%02d_events.tsv' % (sub_name, run_file_idx)
                run_name = 'ses-action01_task-action_run-%d' % run_file_idx
                # fit design matrix based on trial onset time
                events_raw = pd.read_csv(pjoin(sub_func_path, events_file_name), sep='\t')
                duration = events_raw['duration']
                onset = events_raw['onset'].to_numpy() + begin_dur
                label_tmp = events_raw['trial_type'].to_numpy()
                trial_type = ['category-%03d' % idx for idx in label_tmp]
                stim_names = [_.split('/')[-1] for _ in events_raw['stim_file']]
                # load time series
                dtseries_path = pjoin(result_path, f'{run_name}_Atlas.dtseries.nii')
                dtseries = nib.load(dtseries_path).get_fdata()
                # concantenate all info into cycle params
                dtseries_cycle[n_tr*run_idx:n_tr*(run_idx+1)] = dtseries
                onset_cycle[n_event*run_idx:n_event *(run_idx+1)] = onset + run_idx * n_tr * tr
                duration_cycle[n_event*run_idx:n_event*(run_idx+1)] = duration
                trial_type_cycle.extend(trial_type)
                stim_names_cycle.extend(stim_names)
            # prepare design matrix
            events = pd.DataFrame({'trial_type': trial_type_cycle, 'onset': onset_cycle, 'duration': duration_cycle})
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
            beta = reg.coef_[:, :n_class].transpose(1, 0).astype(np.float32)
            # get residual from model
            time_series_predicted = np.dot(design_matrix.values, reg.coef_.transpose(1, 0))
            residual = dtseries_cycle[:, :59412] - time_series_predicted
            # compute CNR: A/?_?
            sigma_noise = residual.std(axis=0)
            amplitude = beta.mean(axis=0)
            cnr_sum[sub_idx, cycle_idx] = amplitude/sigma_noise
            print('Finish performing CNR in %s %s model %02d in alpha %.6f' %(sub_name, sess_name, cycle_idx+1, alpha))
        # save individual cnr
        cnr_individual = np.zeros((91282))
        cnr_individual[:59412] = cnr_sum[sub_idx].mean(axis=0)
        tmp_path = pjoin(save_indiv_path, f'{sub_name}_cnr.dtseries.nii')
        save2cifti(file_path=tmp_path, data=cnr_individual, brain_models=temp)
    # compute coefficient of variation in CNR
    cnr_cv = np.zeros((91282))
    cnr_sub = cnr_sum.mean(axis=1)
    for voxel_idx in range(59412):
        cnr_voxel = cnr_sub[:, voxel_idx]
        cnr_cv[voxel_idx] = cnr_voxel.std()/cnr_voxel.mean()
    save2cifti(file_path=pjoin(result_path, 'cnr_cv.dtseries.nii'), data=cnr_cv, brain_models=temp)
    # save cnr
    cnr_map = np.zeros((91282))
    cnr_sum = cnr_sum.mean(axis=(0, 1))
    cnr_map[:59412] = cnr_sum
    save2cifti(file_path=cnr_path, data=cnr_map, brain_models=temp)
else:
    cnr = nib.load(cnr_path).get_fdata()
    cnr_sum = np.array(cnr).squeeze()[:59412]

# compute cnr mean value across the whole cerebral cortex
print('CNR Mean value across the whole cerebral cortex: %.2f'%(cnr_sum.mean()))
# localize visual area voxels and compute cnr in these regions
# load visual area names
roi_assign = pd.read_csv(pjoin(support_path, 'HCP-MMP1_visual-cortex.csv'))
visual_area = roi_assign['area_name'].to_list()
# load reference info
roi_name_path = pjoin(support_path, 'roilbl_mmp.csv')
roi_all_names = pd.read_csv(roi_name_path)
roi_index = sio.loadmat(pjoin(support_path, 'MMP_mpmLR32k.mat'))['glasser_MMP']  # 1x59412
# generate roi mask
visual_area_mask = roi_mask(visual_area, roi_all_names, roi_index)
cnr_visual_area = cnr_sum[visual_area_mask]
print('CNR Mean value across the visual area: %.2f'%(cnr_visual_area.mean()))

# compute individual cnr value across the whole brain and visual area cortex
cnr_whole_brain_indiv, cnr_visual_area_indiv = np.zeros((30)), np.zeros((30))
for sub_idx, sub_name in enumerate(sub_names):
    tmp_path = pjoin(save_indiv_path, f'{sub_name}_cnr.dtseries.nii')
    cnr_individual = nib.load(tmp_path).get_fdata()
    cnr_individual = np.array(cnr_individual).squeeze()[:59412]
    # compute individual specific values
    cnr_whole_brain_indiv[sub_idx] = cnr_individual.mean()
    cnr_visual_area_indiv[sub_idx] = cnr_individual[visual_area_mask].mean()
    print('CNR value in %s: whole brain: %.2f; visual area: %.2f'%(sub_name, 
            cnr_individual.mean(), cnr_individual[visual_area_mask].mean()))