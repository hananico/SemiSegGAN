. CONFIG

ID=1
GPU=2
NC=4
CLS=0.0
INT=1.0
NGF=128
NDF=64

#NGF=512
#NDF=384

display_id=10${ID} \
gpu=${GPU} \
dataset="eeg" \
name="eeg_img_nc${NC}_cls${CLS}_int${INT}_ngf${NGF}_ndf${NDF}" \
cls_weight=${CLS} \
interp_weight=${INT} \
interp_type=1 \
niter=1000 \
nz=100 \
lr_decay=0.5 \
decay_every=1000 \
img_dir=/home/student/EEG2/new_data/EEGClassesAll \
feat_dir=/home/student/icml2016/eeg_h5files \
data_root=${CUB_META_DIR} \
classnames=${CUB_META_DIR}/allclasses.txt \
trainids=${CUB_META_DIR}/trainvalids.txt \
batchset=/home/student/icml2016/eeg_pairs_batch.txt \
init_t=${CUB_NET_TXT} \
nThreads=1 \
checkpoint_dir=/home/student/icml2016/checkpoints \
numCaption=${NC} \
print_every=4 \
save_every=50 \
replicate=0 \
use_cudnn=1 \
ngf=${NGF} \
ndf=${NDF} \
th main_eeg2img_cond.lua
#th main_cls2img.lua

