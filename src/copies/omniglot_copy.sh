cd ../
mkdir -p job_outputs/experiments/cleaned/omniglot/default/
mkdir -p job_outputs/experiments/cleaned/omniglot/alex/
cp job_outputs/experiments/"controller:default_dataset:omniglot_nclasses:5_seqlength:35_pretrained:False_classdifficulty:all_memsize:128_memvector:40_rnn_size:200"/accuracy95 job_outputs/experiments/cleaned/omniglot/default/
cp job_outputs/experiments/"controller:alex_dataset:omniglot_nclasses:5_seqlength:35_pretrained:False_classdifficulty:all_memsize:128_memvector:40_rnn_size:200"/accuracy95 job_outputs/experiments/cleaned/omniglot/alex/
cd copies
