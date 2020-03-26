python train_softmax_tuned.py --random_crop --random_flip --random_black_patches --learning_rate 0.00001 --max_nrof_epochs 1 --save_every 1
python train_softmax_tuned.py --random_crop --random_flip --random_black_patches --pretrained_model ../models/20191015 --learning_rate 0.005 --max_nrof_epochs 10 --save_every 10
python train_softmax_tuned.py --random_crop --random_flip --random_black_patches --pretrained_model ../models/20191015 --learning_rate 0.001 --max_nrof_epochs 20 --save_every 10
python train_softmax_tuned.py --random_crop --random_flip --random_black_patches --pretrained_model ../models/20191015 --learning_rate 0.0005 --max_nrof_epochs 40 --save_every 10
python train_softmax_tuned.py --random_crop --random_flip --random_black_patches --pretrained_model ../models/20191015 --learning_rate 0.00005 --max_nrof_epochs 55 --save_every 5



