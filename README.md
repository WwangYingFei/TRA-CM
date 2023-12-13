# TRA-CM

quick start:
--train
--optim
adam
--eval_freq
100
--check_point
100
--dataset
demo
--learning_rate
0.001
--lr_decay
0.5
--weight_decay
1e-5
--dropout_rate
0.4
--batch_size
64
--num_steps
20000
--embed_size
128
--hidden_size
256
--patience
7
--combine
exp_mul
--model_dir
.\outputs\models
--result_dir
.\outputs\results
--summary_dir
.\outputs\summary
--log_dir
.\outputs\log

data:
The training, validation, and testing data for TianGong-ST and TREC2014 are all stored in my repository.
