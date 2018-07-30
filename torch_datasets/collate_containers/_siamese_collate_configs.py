from __future__ import division
from copy import deepcopy

from ..utils.collections import AttrDict

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

_c.image_width     = 224
_c.image_height    = 224
_c.stretch_to_fill = True

_c.allow_transform = False
_c.min_rotation    = -0.1
_c.max_rotation    =  0.1
_c.min_translation = (-0.1, -0.1)
_c.max_translation = ( 0.1,  0.1)
_c.min_shear       = -0.1
_c.max_shear       =  0.1
_c.min_scaling     = (0.9, 0.9)
_c.max_scaling     = (1.1, 1.1)
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
