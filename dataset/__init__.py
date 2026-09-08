from .data_loader import TrackedData,TrackedData_infer

def build_dataset(data_cfg, split,):

    return TrackedData(data_cfg, split)