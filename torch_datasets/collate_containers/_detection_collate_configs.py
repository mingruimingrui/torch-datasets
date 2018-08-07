from __future__ import division
from copy import deepcopy

from ..utils.collections import AttrDict

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

_c.allow_image_resize = True
_c.image_min_side     = 800
_c.image_max_side     = 1333

_c.allow_transform = False
_c.min_rotation    = -0.05
_c.max_rotation    =  0.05
_c.min_translation = (-0.05, -0.05)
_c.max_translation = ( 0.05,  0.05)
_c.min_shear       = -0.05
_c.max_shear       =  0.05
_c.min_scaling     = (0.95, 0.95)
_c.max_scaling     = (1.05, 1.05)
_c.flip_x_chance   = 0.5
_c.flip_y_chance   = 0.0

################################################################################
#### End of configurable parameters

# Set default configs to be immutable
_c.immutable(True)

def make_configs(**kwargs):
    configs = deepcopy(_c)
    configs.immutable(False)

    # Update default configs with user provided ones
    for arg, value in kwargs.items():
        configs[arg] = value

    configs.immutable(True)

    return configs
