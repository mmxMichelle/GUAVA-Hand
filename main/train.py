import os
import torch
import argparse
import lightning
from datetime import datetime

import shutil
from dataset import build_dataset
from main.trainer import Trainer
from models.UbodyAvatar import Ubody_Gaussian_inferer,GaussianRenderer,configure_optimizers
from utils.general_utils import (
    ConfigDict, device_parser,
    calc_parameters,add_extra_cfgs
)


def train( config_name, base_model, devices, debug=False):
    meta_cfg = ConfigDict(
        model_config_path=os.path.join('configs', f'{config_name}.yaml')
    )
    meta_cfg = add_extra_cfgs(meta_cfg)
    lightning.fabric.seed_everything(10)
    target_devices = device_parser(devices)
    init_iter = 1
    print(str(meta_cfg))
    # setup model and optimizer
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    render_model = GaussianRenderer(meta_cfg.MODEL)

    optimizer, scheduler =configure_optimizers(infer_model,meta_cfg.OPTIMIZE,render_model)
    op_para_num, all_para_num = calc_parameters([infer_model,render_model])
    print('Number of parameters: {:.2f}M / {:.2f}M.'.format(op_para_num/1000000, all_para_num/1000000))
    if base_model is not None:
        assert os.path.exists(base_model), f'Base model not found: {base_model}.'
        _state=torch.load(base_model, map_location='cpu', weights_only=True)
        infer_model.load_state_dict(_state['model'], strict=False)
        render_model.load_state_dict(_state['render_model'], strict=False)
        init_iter=_state['global_iter']
        print('Load base model from: {}.'.format(base_model))

    # load dataset
    train_dataset = build_dataset(data_cfg=meta_cfg, split='train')
    val_dataset = build_dataset(data_cfg=meta_cfg, split='valid')

    print(f'Train Dataset: {len(train_dataset)}, Val Dataset: {len(val_dataset)}.')
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    if debug:
        train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=meta_cfg.TRAIN.batch_size, num_workers=0, shuffle=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=0, shuffle=False,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=meta_cfg.TRAIN.batch_size, num_workers=1, shuffle=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=1, shuffle=False,
        )

        _dump_path=os.path.join('outputs', meta_cfg.TRAIN.exp_str, timestamp)
        os.makedirs(_dump_path, exist_ok=True)
        shutil.copy(os.path.join('configs', f'{config_name}.yaml'), os.path.join(_dump_path, 'config.yaml'))

    lightning_trainer = Trainer(
        meta_cfg, infer_model,render_model, optimizer, scheduler,
        train_dataloader, val_dataloader,
        devices=target_devices, debug=debug,timestamp=timestamp,

    )
    lightning_trainer.run_fit(init_iter=init_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--basemodel', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    torch.set_float32_matmul_precision('high')
    train(args.config_name, args.basemodel, args.devices, args.debug)