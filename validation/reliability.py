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
save_indiv_path = pjoin(result_path, 'brain_map_individual')
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
# template for saving dtseries
temp = nib.load(pjoin(support_path, 'template.dtseries.nii'))

# start computing reliability 
# for alpha in alphas:
reliability_sum = np.zeros((n_sub, 59412))
for sub_idx, sub_name in enumerate(sub_names):
    # extract beta in each cycle results
    beta_sub = np.zeros((4, 180, 59412))
    for cycle_idx in range(n_cycle):
        beta_cycle = nib.load(pjoin(ciftify_path, sub_name, 'results', f'ses-action01_task-action_cycle-{cycle_idx+1}_beta.dscalar.nii')).get_fdata()
        # beta_cycle = nib.load(pjoin(beta_path, f'alpha-{alpha}', f'{sub_name}_cycle-{cycle_idx+1}_beta.dscalar.nii')).get_fdata()
        # scale data
        scaler = StandardScaler()
        beta_sub[cycle_idx] = scaler.fit_transform(beta_cycle)
    # generate odd cycle and even cycle pattern
    odd_cycle_pattern = beta_sub[[0, 2], :, :].mean(axis=0)
    even_cycle_pattern = beta_sub[[1, 3], :, :].mean(axis=0)
    # iterate on voxel to generate reliability
    for voxel_idx in range(beta_sub.shape[-1]):
        reliability_sum[sub_idx, voxel_idx] = pearsonr(odd_cycle_pattern[:, voxel_idx], even_cycle_pattern[:, voxel_idx])[0]
    # save individual cnr
    reliability_individual = np.zeros((91282))
    reliability_individual[:59412] = reliability_sum[sub_idx]
    tmp_path = pjoin(save_indiv_path, f'{sub_name}_reliability.dtseries.nii')
    save2cifti(file_path=tmp_path, data=reliability_individual, brain_models=temp)
    print(f'Finish computing reliability in {sub_name} in alpha {alpha}')

# compute coefficient of variation in reliability
reliability_cv = np.zeros((91282))
for voxel_idx in range(59412):
    cnr_voxel = reliability_sum[:, voxel_idx]
    reliability_cv[voxel_idx] = cnr_voxel.std()/cnr_voxel.mean()
save2cifti(file_path=pjoin(result_path, 'reliability_cv.dtseries.nii'), data=reliability_cv, brain_models=temp)

# save reliability map
reliability_map = np.zeros((91282))
reliability_map[:59412] = reliability_sum.mean(axis=0)
reliability_path = pjoin(result_path, f'reliability.dtseries.nii')
# reliability_path = pjoin(result_path, 'result_in_different_alpha', f'reliability_alpha-{alpha}.dtseries.nii')
save2cifti(file_path=reliability_path, data=reliability_map, brain_models=temp)