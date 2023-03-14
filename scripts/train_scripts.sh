# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --device_idx 7 --lr 1e-5
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --warmup --alias cosine_scheduler_test --device_idx 7 --lr 1e-5 --optimizer adam --scheduler cosine
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --warmup --alias cosine_scheduler_test --device_idx 6 --lr 5e-6 --optimizer adam --scheduler cosine
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --warmup --alias cosine_scheduler_test --device_idx 4 --lr 5e-6 --optimizer adam --scheduler cosine --warmup
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --device_idx 3 --lr 1e-5 --alias MHSA_LM_repaired
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --device_idx 7 --lr 5e-5 --alias MHSA_LM_test
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --alias MHSA_LAYER_infer_rep  --device_idx 7 --lr 1e-4 --optimizer adam --batch_size 64
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --alias MHSA_LAYER_infer_rep  --device_idx 6 --lr 1e-6 --optimizer adam --batch_size 64
# python /home/cz/bs/rt_torch/train.py --alias rt_torch_vanilla  --device_idx 7 --lr 1e-5 --optimizer adam --batch_size 64
# python /home/cz/bs/rt_torch/train.py --alias rt_torch_vanilla  --device_idx 6 --lr 1e-5 --optimizer adam --batch_size 96
# python /home/cz/bs/rt_torch/train.py --alias LT-RT1  --device_idx 7 --lr 1e-5 --optimizer adam --batch_size 96
# python /home/cz/bs/rt_torch/train.py --alias NO-STACK  --device_idx 6 --lr 1e-5 --optimizer adam --batch_size 16
# python /home/cz/bs/rt_torch/train.py --lr 1e-3 --lr_min 1e-4 --device_idx 3 --optimizer adam --batch_size 96 --loader_shuffle --scheduler cosine  --alias LR-RT1-COSINE
# python /home/cz/bs/rt_torch/train.py --alias LT-RT1-debug  --device_idx 7 --lr 1e-5 --optimizer adam --batch_size 96 --test_interval 1 --save_interval 1
python /home/cz/bs/rt_torch/train.py --alias develop --depth 8 --key_dim 128 --train-iters 500000 --test-iters 100 --test-interval 2500 --save-interval 2500 --scheduler cosine --loader_bs 1 --eval-eps 10 --eval-timeout 100 --device_idx 7 --lr_t 1 --lr_eff 1 --lr 1e-5 --min_lr 1e-6
python /home/cz/bs/rt_torch/train.py --alias develop --depth 8 --key_dim 128 --train-iters 500000 --test-iters 100 --test-interval 2500 --save-interval 2500 --scheduler cosine --loader_bs 1 --eval-eps 10 --eval-timeout 100 --device_idx 0 --lr_t 1 --lr_eff 1 --lr 1e-4 --min_lr 1e-5