. CONFIG

checkpoint_dir=/home/student/icml2016/checkpoints \
net_gen=/seg_img_nc4_cls0.9_int1.0_ngf512_ndf384_100_net_G.t7 \
queries=/home/student/icml2016/test.txt  \
base_queries_dir=/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG \
feat_dir=/home/student/icml2016/VGG_fc7 \
dataset=seg \
th img2seg_demo.lua

