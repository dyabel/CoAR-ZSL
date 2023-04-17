#! /bin/bash
CURPATH=$(cd "$(dirname "$0")"; pwd)
echo $CURPATH
cd $CURPATH
pip install timm
t=0.1
k=0.7
h=10.
# j=999
j=129
i=0.1
alpha=0.5
lp=0.1
lc=0.8
la=1.
beta=0.
scale=25
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config-file config/ZSL/cub_4w_2s.yaml --margin ${i} --prefix trans_triplet_ --temp ${t} --seed ${j} --gamma ${k} --atten_thr ${h} --lambda_proto ${lp} --lambda_att ${la} --lambda_cont ${lc} --alpha ${alpha} --beta ${beta} --scale ${scale} --output_dir "checkpoints/cub_4w_2s/trans/debug"
