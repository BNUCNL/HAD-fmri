import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from had_utils import save2cifti
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# define path
# make sure the dataset_path are modified based on your personal dataset downloading directory
dataset_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/data_upload/HAD'
ciftify_path = f'{dataset_path}/derivatives/ciftify'
nifti_path = f'{dataset_path}'
support_path = './support_files'
result_path = './results'
beta_path = pjoin(result_path, 'beta')
# change to path of current file
os.chdir(os.path.dirname(__file__))

# prepare params
sub_names = ['sub-%02d'%(i+1) for i in range(30)]
# sub_names = ['sub-01']
n_cycle = 4
n_sub = len(sub_names)
# alphas = [eval('1e-%d'%level) for level in np.linspace(0,2,3,dtype=int)]
alpha = 0.1

# for alpha in alphas:
# prepare beta
beta_sum = np.zeros((n_sub, 180, 59412))
for sub_idx, sub_name in enumerate(sub_names):
    # extract beta in each cycle results
    beta_sub = np.zeros((4, 180, 59412))
    for cycle_idx in range(n_cycle):
        beta_sub[cycle_idx] = nib.load(pjoin(ciftify_path, sub_name, 'results', f'ses-action01_task-action_cycle-{cycle_idx+1}_beta.dscalar.nii')).get_fdata()
        # beta_sub[cycle_idx] = nib.load(pjoin(save_beta_path, f'alpha-{alpha}', f'{sub_name}_cycle-{cycle_idx+1}_beta.dscalar.nii')).get_fdata()
    beta_sub = beta_sub.mean(axis=0)
    # scale data
    scaler = StandardScaler()
    beta_sum[sub_idx] = scaler.fit_transform(beta_sub)
    print(f'Finish loading beta: {sub_name}')
# compute ISC
isc_sum = np.zeros((n_sub, 59412))
for voxel_idx in range(beta_sum.shape[-1]):
    voxel_pattern = beta_sum[:, :, voxel_idx]
    # ISC was computed by the correlation of each per participant's response profile with the mean pattern of remaining n-1 participants
    for sub_idx in range(n_sub):
        target_pattern = voxel_pattern[sub_idx]
        mean_pattern = voxel_pattern[np.delete(np.arange(n_sub), sub_idx)].mean(axis=0)
        isc_sum[sub_idx, voxel_idx] = pearsonr(target_pattern, mean_pattern)[0]
    print('Finish computing ISC in voxel:%05d in alpha %.3f'%(voxel_idx+1, alpha))
# save ISC map
temp = nib.load(pjoin(support_path, 'template.dtseries.nii'))
isc_map = np.zeros((91282))
isc_map[:59412] = isc_sum.mean(axis=0)
isc_path = pjoin(result_path, f'isc_alpha.dtseries.nii')
# isc_path = pjoin(result_path, 'result_in_different_alpha', f'isc_alpha-{alpha}.dtseries.nii')
save2cifti(file_path=isc_path, data=isc_map, brain_models=temp)