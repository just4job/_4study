python ./train_Struct2GO.py -labels_num 309 -branch 'mf' -batch_size 64

python ./train_Struct2GO.py -labels_num 311 -branch 'cc' -batch_size 64

python ./train_Struct2GO.py -labels_num 713 -branch 'bp' -batch_size 64



python ./train_struct2GO2.py -labels_num 273 -branch 'mf' -batch_size 64

python ./train_struct2GO2.py -labels_num 809 -branch 'bp' -batch_size 64

python ./train_struct2GO2.py -labels_num 298 -branch 'cc' -batch_size 64



python ./train_Struct2GO2.py -labels_num 308 -branch 'mf' -batch_size 64

python ./train_Struct2GO2.py -labels_num 310 -branch 'cc' -batch_size 64

python ./train_Struct2GO2.py -labels_num 713 -branch 'bp' -batch_size 64


python ./train_Struct2GO2.py -labels_num 308 -branch 'mf' -batch_size 32 -dropout 0.2 

python ./train_Struct2GO2.py -labels_num 310 -branch 'cc' -batch_size 32 -dropout 0.2

python ./train_Struct2GO2.py -labels_num 713 -branch 'bp' -batch_size 32 -dropout 0.1