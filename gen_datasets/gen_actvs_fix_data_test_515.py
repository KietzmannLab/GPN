# Extracting the coco-test-515 dataset and coco-fixation-test-515 dataset for local assessment

import argparse
parser = argparse.ArgumentParser(description='Obtaining hyps')
parser.add_argument('--dva_dataset', type=str, default='NSD')
args = parser.parse_args()

import h5py
import numpy as np

idxs_use = np.load(f'/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds{515}.npy') # selecting the special 515 images from the NSD dataset which will be the test set for GPN

# Extracting coco-test-515 dataset
original_data = h5py.File('/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze_16_fixations.h5', 'r')
images = original_data['test']['data'][idxs_use][()]
embeddings = original_data['test']['all_mpnet_base_v2_mean_embeddings'][idxs_use][()]
# Save the extracted dataset
output_data = h5py.File('/share/klab/datasets/GPN/coco-test-515.h5', 'w')
output_data.create_dataset('images', data=images)
output_data.create_dataset('embeddings', data=embeddings)
output_data.close()
original_data.close()
print('Done extracting coco-test-515 dataset')

# Extracting coco-fixation-test-515 dataset given parameters
original_data = h5py.File(f'/share/klab/datasets/GPN/coco_{args.dva_dataset}_dg3fix{91}_r50v{1}ap_{7}fix_test.h5', 'r')
dg3_fix_actvs = original_data['test']['dg3_fix_actvs'][idxs_use][()]
next_fix_coords = original_data['test']['next_fix_coords'][idxs_use][()]
next_fix_rel_coords = original_data['test']['next_fix_rel_coords'][idxs_use][()]
mpnet_embeddings = original_data['test']['mpnet_embeddings'][idxs_use][()]
full_image_actvs = original_data['test']['full_image_actvs'][idxs_use][()]
# Save the extracted dataset
output_data = h5py.File(f'/share/klab/datasets/GPN/coco_{args.dva_dataset}_dg3fix{91}_r50v{1}ap_{7}fix_test_515.h5', 'w')
output_data.create_dataset('dg3_fix_actvs', data=dg3_fix_actvs)
output_data.create_dataset('next_fix_coords', data=next_fix_coords)
output_data.create_dataset('next_fix_rel_coords', data=next_fix_rel_coords)
output_data.create_dataset('mpnet_embeddings', data=mpnet_embeddings)
output_data.create_dataset('full_image_actvs', data=full_image_actvs)
output_data.close()
original_data.close()
print(f'Done extracting coco_{args.dva_dataset}_dg3fix{91}_r50v{1}ap_{7}fix_test_515 dataset')

