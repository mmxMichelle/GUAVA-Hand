
import os,math,yaml,time,random,glob
from datetime import datetime
import logging
from zoneinfo import ZoneInfo
from omegaconf import OmegaConf
import colored
from rich.progress import BarColumn, Progress, ProgressColumn, Text, TimeElapsedColumn, TimeRemainingColumn, filesize
from tqdm.std import tqdm as std_tqdm
import numpy as np
import torch

class ConfigDict(dict):
    def __init__(self, model_config_path=None, data_config_path=None, init_dict=None):
        if init_dict is None:
            config_dict = read_config(model_config_path)
            if data_config_path is not None:
                dataset_dict = read_config(data_config_path)
                merge_a_into_b(dataset_dict, config_dict)
            # set output path
            experiment_string = '{}_{}'.format(
                config_dict['MODEL']['NAME'], config_dict['DATASET']['NAME']
            )
            timeInTokyo = datetime.now()
            timeInTokyo = timeInTokyo.astimezone(ZoneInfo('Asia/Tokyo'))
            time_string = timeInTokyo.strftime("%b%d_%H%M_")+ \
                        "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
            config_dict['TRAIN']['EXP_STR'] = experiment_string
            config_dict['TRAIN']['TIME_STR'] = time_string
        else:
            config_dict = init_dict
        super().__init__(config_dict)
        self._dot_config = OmegaConf.create(dict(self))
        OmegaConf.set_readonly(self._dot_config, True)

    def __getattr__(self, name):
        if name == '_dump':
            return dict(self)
        if name == '_raw_string':
            import re
            ansi_escape = re.compile(r'''
                \x1B  # ESC
                (?:   # 7-bit C1 Fe (except CSI)
                    [@-Z\\-_]
                |     # or [ for CSI, followed by a control sequence
                    \[
                    [0-?]*  # Parameter bytes
                    [ -/]*  # Intermediate bytes
                    [@-~]   # Final byte
                )
            ''', re.VERBOSE)
            result = '\n' + ansi_escape.sub('', pretty_dict(self))
            return result
        return getattr(self._dot_config, name)

    def __str__(self, ):
        return pretty_dict(self)

    def update(self, key, value):
        OmegaConf.set_readonly(self._dot_config, False)
        self._dot_config[key] = value
        self[key] = value
        OmegaConf.set_readonly(self._dot_config, True)

def add_extra_cfgs(meta_cfg):
    # add extra config options
    OmegaConf.set_readonly(meta_cfg, False)

    if 'with_smplx_gaussian' not in meta_cfg.MODEL.keys():
        meta_cfg.MODEL['with_smplx_gaussian'] = True

    OmegaConf.set_readonly(meta_cfg, True)
    return meta_cfg

def read_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found.")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v

def pretty_dict(input_dict, indent=0, highlight_keys=[]):
    out_line = ""
    tab = "    "
    for key, value in input_dict.items():
        if key in highlight_keys:
            out_line += tab * indent + colored.stylize(str(key), colored.fg(1))
        else:
            out_line += tab * indent + colored.stylize(str(key), colored.fg(2))
        if isinstance(value, dict):
            out_line += ':\n'
            out_line += pretty_dict(value, indent+1, highlight_keys)
        else:
            if key in highlight_keys:
                out_line += ":" + "\t" + colored.stylize(str(value), colored.fg(1)) + '\n'
            else:
                out_line += ":" + "\t" + colored.stylize(str(value), colored.fg(2)) + '\n'
    if indent == 0:
        max_length = 0
        for line in out_line.split('\n'):
            max_length = max(max_length, len(line.split('\t')[0]))
        max_length += 4
        aligned_line = ""
        for line in out_line.split('\n'):
            if '\t' in line:
                aligned_number = max_length - len(line.split('\t')[0])
                line = line.replace('\t',  aligned_number * ' ')
            aligned_line += line+'\n'
        return aligned_line[:-2]
    return out_line


class rtqdm(std_tqdm):  # pragma: no cover
    """Experimental rich.progress GUI version of tqdm!"""
    # TODO: @classmethod: write()?
    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        progress  : tuple, optional
            arguments for `rich.progress.Progress()`.
        options  : dict, optional
            keyword arguments for `rich.progress.Progress()`.
        """
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        # convert disable = None to False
        kwargs['disable'] = bool(kwargs.get('disable', False))
        progress = kwargs.pop('progress', None)
        options = kwargs.pop('options', {}).copy()
        super(rtqdm, self).__init__(*args, **kwargs)

        if self.disable:
            return

        d = self.format_dict
        if progress is None:
            progress = (
                "[progress.description]"
                "[progress.percentage]{task.percentage:>4.0f}%",
                BarColumn(bar_width=66),
                FractionColumn(unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
                "[",
                    TimeElapsedColumn(), "<", TimeRemainingColumn(), ",",
                    RateColumn(unit=d['unit'], unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
                    "{task.description}",
                "]",
            )
        options.setdefault('transient', not self.leave)
        self._prog = Progress(*progress, **options)
        self._prog.__enter__()
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        if self.disable:
            return
        super(rtqdm, self).close()
        self._prog.__exit__(None, None, None)

    def clear(self, *_, **__):
        pass

    def set_postfix(self, desc):

        desc_str = ", "+" , ".join([
            colored.stylize(str(f"{k}"), colored.fg(3)) + " = " +
            colored.stylize(str(f"{v}"), colored.fg(4))
            for k, v in desc.items()]
        )
        self.desc = desc_str
        self.display()

    def display(self, *_, **__):
        if not hasattr(self, '_prog'):
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc)

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_prog'):
            self._prog.reset(total=total)
        super(rtqdm, self).reset(total=total)

class FractionColumn(ProgressColumn):
    """Renders completed/total, e.g. '0.5/2.3 G'."""
    def __init__(self, unit_scale=False, unit_divisor=1000):
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Calculate common unit for completed and total."""
        completed = int(task.completed)
        total = int(task.total)
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                total,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(total, [""], 1)
        precision = 0 if unit == 1 else 1
        return Text(
            f"{completed/unit:,.{precision}f}/{total/unit:,.{precision}f} {suffix}",
            style="progress.download")


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""
    def __init__(self, unit="", unit_scale=False, unit_divisor=1000):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"? {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        precision = 0 if unit == 1 else 1
        return Text(f"{speed/unit:,.{precision}f} {suffix}{self.unit}/s",
                    style="progress.data.speed")

def device_parser(str_device):
    def parser_dash(str_device):
        device_id = str_device.split('-')
        device_id = [i for i in range(int(device_id[0]), int(device_id[-1])+1)]
        return device_id
    if 'cpu' in str_device:
        device_id = ['cpu']
    else:
        device_id = str_device.split(',')
        device_id = [parser_dash(i) for i in device_id]
    res = []
    for i in device_id:
        res += i
    return res

def device_parser(str_device):
    def parser_dash(str_device):
        device_id = str_device.split('-')
        device_id = [i for i in range(int(device_id[0]), int(device_id[-1])+1)]
        return device_id
    if 'cpu' in str_device:
        device_id = ['cpu']
    else:
        device_id = str_device.split(',')
        device_id = [parser_dash(i) for i in device_id]
    res = []
    for i in device_id:
        res += i
    return res

def calc_parameters(models):
    op_para_nums=0
    all_para_nums=0
    for model in models:
        op_para_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_para_num = sum(p.numel() for p in model.parameters())
        op_para_nums += op_para_num
        all_para_nums += all_para_num
    return op_para_nums, all_para_nums

def biuld_logger(log_path, name='test_logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def find_pt_file(base_path, prefix):
    pt_files = glob.glob(os.path.join(base_path, f"{prefix}*.pt"))
    if pt_files:
        return max(pt_files, key=os.path.getmtime)
    return None

def to8b(img):
    return (255 * np.clip(img, 0, 1)).astype(np.uint8)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))