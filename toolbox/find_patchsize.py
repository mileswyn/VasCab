# 用于找到相对最优的fingerprint
import numpy as np
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.network_architecture.generic_UNet import Generic_UNet
from copy import deepcopy

new_median_shape = np.array([200, 360, 360])
current_spacing = np.array([0.633, 0.487, 0.487])

# new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
# dataset_num_voxels = np.prod(new_median_shape) * num_cases

# the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
# input_patch_size = new_median_shape

# compute how many voxels are one mm
input_patch_size = 1 / np.array(current_spacing)

# normalize voxels per mm
input_patch_size /= input_patch_size.mean()

# create an isotropic patch of size 512x512x512mm
input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
input_patch_size = np.round(input_patch_size).astype(int)

# clip it to the median shape of the dataset because patches larger then that make not much sense
input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]

network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, input_patch_size,
                                                        4,
                                                        999)

# we compute as if we were using only 30 feature maps. We can do that because fp16 training is the standard
# now. That frees up some space. The decision to go with 32 is solely due to the speedup we get (non-multiples
# of 8 are not supported in nvidia amp)
ref = 520000000 * 32 / \
        30
here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                    32,
                                                    320, 1,
                                                    2,
                                                    pool_op_kernel_sizes, conv_per_stage=2)
while here > ref:
    axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

    tmp = deepcopy(new_shp)
    tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
    _, _, _, _, shape_must_be_divisible_by_new = \
        get_pool_and_conv_props(current_spacing, tmp,
                                4,
                                999,
                                )
    new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

    # we have to recompute numpool now:
    network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
    shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, new_shp,
                                                            4,
                                                            999,
                                                            )

    here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                        32,
                                                        320, 1,
                                                        2, pool_op_kernel_sizes,
                                                        conv_per_stage=2)
    #print(new_shp)
#print(here, ref)

input_patch_size = new_shp
print(input_patch_size)