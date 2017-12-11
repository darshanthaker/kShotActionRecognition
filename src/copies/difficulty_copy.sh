cd ../
mkdir -p job_outputs/experiments/cleaned/difficulty/easy/
mkdir -p job_outputs/experiments/cleaned/difficulty/medium/
mkdir -p job_outputs/experiments/cleaned/difficulty/hard/
mkdir -p job_outputs/experiments/cleaned/difficulty/all/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:5_seqlength:35_pretrained:False_classdifficulty:all_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/difficulty/all/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:5_seqlength:35_pretrained:False_classdifficulty:easy_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/difficulty/easy/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:5_seqlength:35_pretrained:False_classdifficulty:medium_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/difficulty/medium/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:5_seqlength:35_pretrained:False_classdifficulty:hard_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/difficulty/hard/
cd copies
