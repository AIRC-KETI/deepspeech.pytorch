from __future__ import print_function

import fnmatch
import io
import os

import torch
from tqdm import tqdm
import subprocess


def create_manifest(data_path, output_path, min_duration=None, max_duration=None):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)
    with io.FileIO(output_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')


def order_and_prune_files(file_paths, min_duration, max_duration):
    print("Sorting manifests...")
    duration_file_paths = [(path, float(subprocess.check_output(
        ['soxi -D \"%s\"' % path.strip()], shell=True))) for path in file_paths]
    if min_duration and max_duration:
        print("Pruning manifests between %d and %d seconds" % (min_duration, max_duration))
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               min_duration <= duration <= max_duration]

    def func(element):
        return element[1]

    duration_file_paths.sort(key=func)
    return [x[0] for x in duration_file_paths]  # Remove durations


def BN_convert_float(module):
    """
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return BN_convert_float(network.half())


def set_grad(param_copy, param, scale_factor=1):
    for p_optim, p_model in zip(param_copy, param):
        p_optim.grad = p_model.grad.float() / (1 / scale_factor)
