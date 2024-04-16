# python evaluate.py  'maml' \
#                     --use_cuda \
#                     --gpu 0 \
#                     --folder './data/' \
#                     --dataset 'mini' \
#                     --seed 1 \
#                     --hidden-size 64 \
#                     --embedding-size 1600 \
#                     --nb-test-task-update 10 \
#                     --step-size 0.01 \
#                     --num-shots 1 \
#                     --num-ways 5 \
#                     --save \
#                     --output-folder './checkpoints/' \
#                     -f "mini_maml_5way_1shot_2nd_order_seed1.th" \
#                     --num-test-ep 600 \
#                     --test-batch-size 4

python evaluate.py 'protonet' \
                    --use_cuda \
                    --gpu 0 \
                    --folder './data/' \
                    --dataset 'mini' \
                    --seed 1 \
                    --hidden-size 64 \
                    --num-shots 1 \
                    --num-ways 5 \
                    --save \
                    --output-folder './checkpoints/' \
                    -f "mini_protonet_5way_1shot_norm_seed1.th" \
                    --num-test-ep 600 \
                    --test-batch-size 4
