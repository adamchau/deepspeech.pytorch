from __future__ import print_function

import fnmatch
import io
import os
from tqdm import tqdm
import subprocess
import torch.distributed as dist


def create_manifest(data_path, output_path, min_duration=None, max_duration=None):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)
    with io.FileIO(output_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            print(wav_path)
            file.write(sample.encode('utf-8'))
    print('\n')

def create_manifest_aishell2(wav_scp, manifest_file):
    f = open(manifest_file, 'w')
    for line in open(wav_scp, 'r').readlines():
        path = '/home/ydzhao/PycharmProjects/deepspeech.pytorch/dataset/aishell2/iOS/data/'+ line.strip().split('\t')[1]
        label = path.replace('.wav', '.txt')
        sample = path+','+label+'\n'
        f.write(sample)
        print(path)
    f.close()
    print('\n')


def create_manifest_th30hr(data_path, output_path, min_duration=None, max_duration=None):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)
    with io.FileIO(output_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path+'.trn'
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

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

if __name__ == "__main__":
    # data_path = '/home/ydzhao/PycharmProjects/deepspeech.pytorch/dataset/thchs30/data'
    # output_path = './manifest_ths30.txt'
    # data_path = '/home/ydzhao/PycharmProjects/deepspeech.pytorch/dataset/thchs30/data'
    # output_path = './manifest_ths30.txt'
    # create_manifest_th30hr(data_path, output_path)
    # data_path = '/home/ydzhao/PycharmProjects/deepspeech.pytorch/dataset/aishell/data_aishell/wav'
    # output_path = './manifest_aishell1.txt'
    # create_manifest(data_path, output_path)
    data_path = '/home/ydzhao/PycharmProjects/deepspeech.pytorch/dataset/aishell2/iOS/data/wav.scp'
    output_path = '/home/ydzhao/PycharmProjects/deepspeech.pytorch/dataset/aishell2/iOS/data/wav2.scp'
    create_manifest_aishell2(data_path, output_path)
