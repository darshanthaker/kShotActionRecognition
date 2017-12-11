cd ../
mkdir -p job_outputs/experiments/cleaned/classes/n10seq70/
mkdir -p job_outputs/experiments/cleaned/classes/n15seq105/
mkdir -p job_outputs/experiments/cleaned/classes/n25seq175/
mkdir -p job_outputs/experiments/cleaned/classes/n5seq35/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:10_seqlength:70_pretrained:False_classdifficulty:all_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/classes/n10seq70/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:15_seqlength:105_pretrained:False_classdifficulty:all_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/classes/n15seq105/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:25_seqlength:175_pretrained:False_classdifficulty:all_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/classes/n25seq175/
cp job_outputs/experiments/"controller:alex_dataset:kinetics_dynamic_nclasses:5_seqlength:35_pretrained:False_classdifficulty:all_memsize:128_memvector:40_rnn_size:200_img:64"/inter_accuracy44 job_outputs/experiments/cleaned/classes/n5seq35/
cd copies
