# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --device_idx 7 --lr 1e-5
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --warmup --alias cosine_scheduler_test --device_idx 7 --lr 1e-5 --optimizer adam --scheduler cosine
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --warmup --alias cosine_scheduler_test --device_idx 6 --lr 5e-6 --optimizer adam --scheduler cosine
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --warmup --alias cosine_scheduler_test --device_idx 4 --lr 5e-6 --optimizer adam --scheduler cosine --warmup
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --device_idx 3 --lr 1e-5 --alias MHSA_LM_repaired
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --device_idx 7 --lr 5e-5 --alias MHSA_LM_test
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --alias MHSA_LAYER_infer_rep  --device_idx 7 --lr 1e-4 --optimizer adam --batch_size 64
# python /home/cz/bs/robotic-transformer-pytorch/robotic_transformer/train.py --alias MHSA_LAYER_infer_rep  --device_idx 6 --lr 1e-6 --optimizer adam --batch_size 64
# python /home/cz/bs/rt_torch/train.py --alias rt_torch_vanilla  --device_idx 7 --lr 1e-5 --optimizer adam --batch_size 64
# python /home/cz/bs/rt_torch/train.py --alias rt_torch_vanilla  --device_idx 6 --lr 1e-5 --optimizer adam
# python /home/cz/bs/rt_torch/train.py --alias LT-RT1  --device_idx 7 --lr 1e-5 --optimizer adam
# python /home/cz/bs/rt_torch/train.py --alias NO-STACK  --device_idx 6 --lr 1e-5 --optimizer adam --batch_size 16
# python /home/cz/bs/rt_torch/train.py --lr 1e-3 --lr_min 1e-4 --device_idx 3 --optimizer adam --loader_shuffle --scheduler cosine  --alias LR-RT1-COSINE
# python /home/cz/bs/rt_torch/train.py --alias LT-RT1-debug  --device_idx 7 --lr 1e-5 --optimizer adam --test_interval 1 --save_interval 1
python /home/cz/bs/rt_torch/train.py --alias develop --depth 8 --key_dim 128 --scheduler cosine --device_idx 7 --lr 1e-5 --min_lr 1e-6
python /home/cz/bs/rt_torch/train.py --alias develop --depth 8 --key_dim 128 --scheduler cosine --device_idx 0 --lr 1e-4 --min_lr 1e-5

python /home/cz/bs/rt_torch/train.py --alias fusion-single --device_idx 7 --model fusion  --scheduler cosine --lr 1e-4 --min_lr 1e-5 --heads 4 --model_dim 256
python /home/cz/bs/rt_torch/train.py --alias fusion-single --device_idx 6 --model fusion  --scheduler cosine --lr 1e-3 --min_lr 1e-4 --heads 4 --model_dim 256
python /home/cz/bs/rt_torch/train.py --alias fusion-single --device_idx 5 --model fusion  --scheduler cosine --lr 1e-5 --min_lr 1e-6 --heads 4 --model_dim 256

python /home/cz/bs/rt_torch/train.py --alias fusion-single --device_idx 5 --model fusion --scheduler cosine --lr 1e-5 --min_lr 1e-6 --heads 4 --model_dim 256


python /home/cz/bs/rt_torch/train.py --alias vanilla-use --device_idx 7 --model vanilla --scheduler cosine --lr 1e-5 --min_lr 1e-6 --text_encoder "use" --batch_size 160

python /home/cz/bs/rt_torch/train.py --alias fusion-use --device_idx 6 --model fusion --scheduler cosine --lr 1e-5 --min_lr 1e-6 --heads 4 --model_dim 256 --text_encoder "use"

python /home/cz/bs/rt_torch/train.py --alias vanilla-use --device_idx 7 --model vanilla --scheduler cosine --lr 1e-4 --min_lr 1e-5 --text_encoder "use" --batch_size 160

python /home/cz/bs/rt_torch/train.py --alias fusion-use --device_idx 6 --model fusion --scheduler cosine --lr 1e-4 --min_lr 1e-5 --heads 4 --model_dim 256 --text_encoder "use"

python /home/cz/bs/rt_torch/train.py --alias vanilla-use --device_idx 3 --model vanilla --scheduler cosine --lr 1e-3 --min_lr 1e-4 --text_encoder "use" --batch_size 160

python /home/cz/bs/rt_torch/train.py --alias fusion-use --device_idx 2 --model fusion --scheduler cosine --lr 1e-3 --min_lr 1e-4 --heads 4 --model_dim 256 --text_encoder "use"