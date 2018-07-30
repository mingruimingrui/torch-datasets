
import torch


def deepcuda(obj, device_idx=None):
    """ Recursively copies a list, tuple, dict, module or tensor typed object into device memory
    Think of deepcopy but for cuda tensor conversion
    """

    obj_is_tensor = torch.tensor._TensorBase in obj.__class__.__bases__
    obj_is_module = torch.nn.modules.module.Module in obj.__class__.__bases__

    if obj_is_tensor or obj_is_module:
        return obj.cuda()

    elif isinstance(obj, list):
        return [deepcuda(o) for o in obj]

    elif isinstance(obj, tuple):
        return tuple(deepcuda(o) for o in obj)

    elif isinstance(obj, dict):
        return {k: deepcuda(v) for (k, v) in obj.items()}
