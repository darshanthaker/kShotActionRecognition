import os
import sys
import numpy as np
import argparse
import subprocess
from pdb import set_trace
from util import eprint

VALID_N_CLASSES = [5, 15, 25]
VALID_CONTROLLERS = ['alex', 'vgg19']
VALID_CLASS_DIFFICULTIES = ['easy', 'medium', 'hard', 'all']

class ExperimentConfig(object):

  
    def __init__(self, dataset_type='kinetics_dynamic', controller_type='alex', \
            batch_size=16, image_width=128, image_height=128, \
            summary_writer=False, model_saver=False, debug=True, \
            memory_size=128, memory_vector_dim=40, seq_length=100, \
            n_classes=25, class_difficulty='all', use_pretrained=False, \
            num_epoches=1000, rnn_size=200, batches_validation=5, im_normalization=True):
        # Sanity checks.
        if controller_type == 'vgg19':
            assert image_height <= 64 and image_width <= 64 and batch_size <= 8
        assert debug == True
        assert n_classes in VALID_N_CLASSES
        assert controller_type in VALID_CONTROLLERS
        assert class_difficulty in VALID_CLASS_DIFFICULTIES
            
        command = "python3 one_shot_learning.py --dataset_type={} " \
                  "--controller_type={} --batch_size={} " \
                  "--image_width={} --image_height={} --summary_writer={} " \
                  "--model_saver={} --debug={} --memory_size={} " \
                  "--memory_vector_dim={} --seq_length={} " \
                  "--n_classes={} --class_difficulty={} --use_pretrained={} " \
                  "--num_epoches={} --rnn_size={} --batches_validation={} " \
                  "--im_normalization={}"
        command = command.format(dataset_type, controller_type, batch_size, \
                image_width, image_height, summary_writer, model_saver, debug, \
                memory_size, memory_vector_dim, seq_length, n_classes, \
                class_difficulty, use_pretrained, num_epoches, rnn_size, \
                batches_validation, im_normalization)
        self.command = command


exp_to_folder_map = {'al_med': 'difficulty', 'vgg': 'controllers',
                    'no_norm': 'no_norm',
                    'mem128x80': 'memory', 'mem128x20': 'memory', 'mem64x40':'memory', 'mem256x40':'memory',
                    'center_frame': 'inputs',
                    'lstm100': 'lstm', 'lstm1': 'lstm', 'lstm300':'lstm'} 

all_configs = {'difficulty/al_med': ExperimentConfig(class_difficulty='medium'), \
               'controllers/vgg': ExperimentConfig(controller_type='vgg19', \
                                                   image_width=64, image_height=64, \
                                                   batch_size=8),
               'no_norm/no_norm': ExperimentConfig(im_normalization=False),
               'no_norm/norm': ExperimentConfig(im_normalization=True),
               'memory/mem128x80': ExperimentConfig(memory_size=128, memory_vector_dim=80),
               'memory/mem128x20': ExperimentConfig(memory_size=128, memory_vector_dim=20),
               'memory/mem64x40': ExperimentConfig(memory_size=64, memory_vector_dim=40),
               'memory/mem256x40': ExperimentConfig(memory_size=256, memory_vector_dim=40),
               'inputs/center_frame': ExperimentConfig(dataset_type='kinetics_single_frame', seq_length=35, n_classes=5, num_epoches=250),
               'lstm/lstm1': ExperimentConfig(rnn_size=1),
               'lstm/lstm100': ExperimentConfig(rnn_size=100),
               'lstm/lstm300': ExperimentConfig(rnn_size=300),
              }

class ExperimentRunner(object):


    def __init__(self, experiments, duration=4):
        self.experiments = experiments
        self.duration = 4
        for exp in experiments:
            self.run_job(exp)

    def run_job(self, exp):
        out_file_path = os.path.join('job_outputs', \
            exp_to_folder_map[exp], exp + '-%j.out')
        type_and_exp = os.path.join(exp_to_folder_map[exp], exp)
        eprint(type_and_exp)
        config = all_configs[type_and_exp]
        if not os.path.exists('job_files'):
            os.mkdir('job_files')
        job_file = 'job_files/{}_runner.sh'.format(exp)
        if os.path.exists(job_file):
            os.remove(job_file)
        with open(job_file, 'w') as fp:
            print("#!/bin/bash", file=fp)
            print("#", file=fp)
            print("#SBATCH -J {}".format(exp), file=fp)
            print("#SBATCH -o {}".format(out_file_path), file=fp)
            print("#SBATCH -p gpu", file=fp)
            print("#SBATCH -N 1", file=fp)
            print("#SBATCH -n 1", file=fp)
            print("#SBATCH -t {}:00:00".format(self.duration), file=fp)
            print("#SBATCH -A CS381V-Visual-Recogn", file=fp)
            print("", file=fp)
            print("module load gcc/4.9.1 cuda/8.0 cudnn/5.1 "\
                  "python3/3.5.2 tensorflow-gpu/1.0.0", file=fp)
            print(config.command, file=fp)
        eprint("Job File: ", job_file)
        eprint(config.command)
        os.chmod(job_file, 0o755)
        p = subprocess.Popen('sbatch ./{}'.format(job_file), shell=True)
        out, err = p.communicate()
        eprint(err)
        eprint(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=str)
    parser.add_argument('--duration', default=4, type=int)
    args = parser.parse_args()
    if not args.experiments:
        print("0 Experiments passed in!")
        sys.exit(1)
    runner = ExperimentRunner(args.experiments.split())

if __name__=='__main__':
    main()
