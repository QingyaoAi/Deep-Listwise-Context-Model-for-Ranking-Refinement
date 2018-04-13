import os,sys

INTITAL_MODEL = 'MART'
SET_NAME = 'set1'
RANK_CUT = '40'
PROGRAM_PATH = '/net/home/aiqy/Project/LSTM-rank/tmp/'
DATA_PATH = '/mnt/scratch/aiqy/LSTM_rank/working/training_data/'+INTITAL_MODEL+'/'+SET_NAME+'/RANK-CUT_'+str(RANK_CUT) + '/'

command = 'python ' + PROGRAM_PATH + 'main.py --data_dir ' + DATA_PATH
command += ' --steps_per_checkpoint 50'
command += ' --embed_size 0'
command += ' --self_test true'

command += ' --hparams ' + ','.join([
    'learning_rate=0.5',
    'num_heads=3',
    'att_strategy=multi',
    #'loss_func=softRank',
    'l2_loss=0.01',
    'use_residua=false'
])

print(command)
os.system(command)