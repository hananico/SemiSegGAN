. CONFIG

ID=1
GPU=0
NC=4
CLS=0.9
INT=1.0
# NGF=128
# NDF=96

NGF=512
NDF=384

display_id=10${ID} \
gpu=${GPU} \
dataset="seg" \
name="seg_img_nc${NC}_cls${CLS}_int${INT}_ngf${NGF}_ndf${NDF}" \
cls_weight=${CLS} \
interp_weight=${INT} \
interp_type=1 \
niter=1000 \
nz=100 \
lr_decay=0.5 \
decay_every=100 \
img_dir=/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG \
feat_dir=/home/student/icml2016/VGG_fc7 \
data_root=${CUB_META_DIR} \
classnames=${CUB_META_DIR}/allclasses.txt \
trainids=${CUB_META_DIR}/trainvalids.txt \
batchset=/home/student/caffe_orig/caffe-master/data/VOC2012_SEG_AUG/batch2.txt \
init_t=${CUB_NET_TXT} \
nThreads=2 \
checkpoint_dir=/home/student/icml2016/checkpoints \
numCaption=${NC} \
print_every=4 \
save_every=100 \
replicate=0 \
use_cudnn=1 \
ngf=${NGF} \
ndf=${NDF} \
th main_segment.lua


