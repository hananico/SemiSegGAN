. CONFIG

checkpoint_dir=/home/student/icml2016/checkpoints \
net_gen=/eeg_img_nc4_cls0.5_int1.0_ngf128_ndf64_100_net_G.t7 \
queries= /home/student/icml2016/eeg_query.txt \
base_queries_dir= /home/student/icml2016/eeg_h5files \
feat_dir=/home/student/icml2016/VGG_fc7 \
dataset=eeg \
th eeg2img_demo.lua

