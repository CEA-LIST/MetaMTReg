# Original Protonet

python mini_protonet.py --use_cuda \
                        --gpu 0 \
                        --folder './data/' \
                        --download \
                        --dataset 'mini' \
                        --seed 1 \
                        --hidden-size 64 \
                        --num-ways 5 \
                        --num-shots 1 \
                        --save \
                        --output-folder './checkpoints/' \
                        -f "mini_protonet_5way_1shot_seed1.th" \
                        --num-batches 60000 \
                        --num-test-ep 600 \
                        --val-interval 1000 \
                        --batch-size 4

# Normalized Protonet

python mini_protonet.py --use_cuda \
                        --gpu 0 \
                        --folder './data/' \
                        --download \
                        --dataset 'mini' \
                        --seed 1 \
                        --hidden-size 64 \
                        --num-ways 5 \
                        --num-shots 1 \
                        --save \
                        --norm \
                        --output-folder './checkpoints/' \
                        -f "mini_protonet_5way_1shot_norm_seed1.th" \
                        --num-batches 60000 \
                        --num-test-ep 600 \
                        --val-interval 1000 \
                        --batch-size 4