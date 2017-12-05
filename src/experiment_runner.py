import os
import sys
import numpy as np
import argparse
import subprocess
from pdb import set_trace

VALID_N_CLASSES = [5, 15, 25]
VALID_CONTROLLERS = ['alex', 'vgg19']
VALID_CLASS_DIFFICULTIES = ['easy', 'medium', 'hard', 'all']

class ExperimentConfig(object):

  
    def __init__(self, dataset_type='kinetics_dynamic', controller_type='alex', \
            batch_size=16, image_width=128, image_height=128, \
            summary_writer=True, model_saver=False, debug=True, \
            memory_size=128, memory_vector_dim=40, seq_length=100, \
            n_classes=25, class_difficulty='all', use_pretrained=False, \
            num_epoches=1000, rnn_size=200, batches_validation=5):
        # Sanity checks.
        if controller_type == 'vgg19':
            assert image_height <= 64 and image_width <= 64  
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
                  "--num_epoches={} --rnn_size={} --batches_validation={}"
        command = command.format(dataset_type, controller_type, batch_size, \
                image_width, image_height, summary_writer, model_saver, debug, \
                memory_size, memory_vector_dim, seq_length, n_classes, \
                class_difficulty, use_pretrained, num_epoches, rnn_size, \
                batches_validation)
        self.command = command


exp_to_folder_map = {'al_med': 'difficulty'}

all_configs = {'difficulty/al_med': ExperimentConfig(class_difficulty='medium')}

class ExperimentRunner(object):


    def __init__(self, experiments):
        self.experiments = experiments
        for exp in experiments:
            self.run_job(exp)

    def run_job(self, exp):
        out_file_path = os.path.join('job_outputs', \
            exp_to_folder_map[exp], exp + '-%j.out')
        type_and_exp = os.path.join(exp_to_folder_map[exp], exp)
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
            print("#SBATCH -t 4:00:00", file=fp)
            print("#SBATCH -A CS381V-Visual-Recogn", file=fp)
            print("", file=fp)
            print("module load gcc/4.9.1 cuda/8.0 cudnn/5.1 "\
                  " python3/3.5.2 tensorflow-gpu/1.0.0", file=fp)
            print(config.command, file=fp)
        os.chmod(job_file, 0o755)
        p = subprocess.Popen('sbatch ./{}'.format(job_file), shell=True)
        out, err = p.communicate()

def main(experiments):
    set_trace()
    runner = ExperimentRunner(experiments)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=str)
    args = parser.parse_args()
    if not args.experiments:
        print("0 Experiments passed in!")
        sys.exit(1)
    main(args.experiments.split())
