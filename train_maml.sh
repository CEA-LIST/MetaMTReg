# Original MAML

python mini_maml.py --use_cuda \
                    --gpu 0 \
                    --folder './data/' \
                    --download \
                    --dataset 'mini' \
                    --seed 1 \
                    --hidden-size 64 \
                    --embedding-size 1600 \
                    --nb-task-update 5 \
                    --nb-test-task-update 10 \
                    --num-ways 5 \
                    --num-shots 1 \
                    --step-size 0.01 \
                    --save \
                    --output-folder './checkpoints/' \
                    -f "mini_maml_5way_1shot_2nd_order_seed1.th" \
                    --num-batches 60000 \
                    --num-test-ep 600 \
                    --val-interval 1000 \
                    --batch-size 4

# Regularized MAML

python mini_maml.py --use_cuda \
                    --gpu 0 \
                    --folder './data/' \
                    --download \
                    --dataset 'mini' \
                    --seed 1 \
                    --hidden-size 64 \
                    --embedding-size 1600 \
                    --nb-task-update 5 \
                    --nb-test-task-update 10 \
                    --num-ways 5 \
                    --num-shots 1 \
                    --step-size 0.01 \
                    --save \
                    --s_ratio \
                    --s_norm \
                    --output-folder './checkpoints/' \
                    -f "mini_maml_5way_1shot_2nd_order_reg_seed1.th" \
                    --num-batches 60000 \
                    --num-test-ep 600 \
                    --val-interval 1000 \
                    --batch-size 4