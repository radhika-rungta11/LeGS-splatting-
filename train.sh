echo "mipnerf开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=bicycle python train.py -s /data2/ningzhh/data/mipnerf360/bicycle -m output/rl/bicycle -i images_4 --eval --densification_interval 100  --optimizer_type default  --grad_abs_thresh 0.0004
OAR_JOB_ID=flowers python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m output/rl/flowers -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004
OAR_JOB_ID=garden python train.py -s /data2/ningzhh/data/mipnerf360/garden -m output/rl/garden -i images_4 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0002
OAR_JOB_ID=stump python train.py -s /data2/ningzhh/data/mipnerf360/stump -m output/rl/stump -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004
OAR_JOB_ID=treehill python train.py -s /data2/ningzhh/data/mipnerf360/treehill -m output/rl/treehill -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0005
OAR_JOB_ID=room python train.py -s /data2/ningzhh/data/mipnerf360/room -m output/rl/room -i images_2 --eval --densification_interval 100  --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002
OAR_JOB_ID=counter python train.py -s /data2/ningzhh/data/mipnerf360/counter -m output/rl/counter -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0002
OAR_JOB_ID=kitchen python train.py -s /data2/ningzhh/data/mipnerf360/kitchen -m output/rl/kitchen -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001
OAR_JOB_ID=bonsai python train.py -s /data2/ningzhh/data/mipnerf360/bonsai -m output/rl/bonsai -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001
end=$(date +%s)
echo "mipnerf结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "mipnerf耗时: $((end - start)) s"

echo "tanksandtemples开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=truck python train.py -s /data2/ningzhh/data/tanksandtemples/truck -m output/rl/truck --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.04 --grad_abs_thresh 0.0001 --mult 0.7
OAR_JOB_ID=train python train.py -s /data2/ningzhh/data/tanksandtemples/train -m output/rl/train --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.042 --grad_abs_thresh 0.0001 --mult 0.7
end=$(date +%s)
echo "tanksandtemples结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "tanksandtemples耗时: $((end - start)) s"

echo "db开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=drjohnson python train.py -s /data2/ningzhh/data/db/drjohnson -m output/rl/drjohnson --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0002 --mult 0.7
OAR_JOB_ID=playroom python train.py -s /data2/ningzhh/data/db/playroom -m output/rl/playroom --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0015 --grad_abs_thresh 0.0002 --mult 0.7
end=$(date +%s)
echo "db结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "db耗时: $((end - start)) s"


python render.py -m output/rl/bicycle --skip_train
python metrics.py -m output/rl/bicycle

python render.py -m output/rl/flowers --skip_train
python metrics.py -m output/rl/flowers

python render.py -m output/rl/garden --skip_train
python metrics.py -m output/rl/garden

python render.py -m output/rl/stump --skip_train
python metrics.py -m output/rl/stump

python render.py -m output/rl/treehill --skip_train
python metrics.py -m output/rl/treehill

python render.py -m output/rl/room --skip_train
python metrics.py -m output/rl/room

python render.py -m output/rl/counter --skip_train
python metrics.py -m output/rl/counter

python render.py -m output/rl/kitchen --skip_train
python metrics.py -m output/rl/kitchen

python render.py -m output/rl/bonsai --skip_train
python metrics.py -m output/rl/bonsai

python render.py -m output/rl/truck --skip_train --mult 0.7
python metrics.py -m output/rl/truck

python render.py -m output/rl/train --skip_train --mult 0.7
python metrics.py -m output/rl/train

python render.py -m output/rl/drjohnson --skip_train --mult 0.7
python metrics.py -m output/rl/drjohnson

python render.py -m output/rl/playroom --skip_train --mult 0.7
python metrics.py -m output/rl/playroom

python report_results.py
