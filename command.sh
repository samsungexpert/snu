CUDA_VISIBLE_DEVICES=0 python mytrain_tf.py --model_name=resnet_ed
CUDA_VISIBLE_DEVICES=1 python mytrain_tf.py --model_name=resnet_flat
CUDA_VISIBLE_DEVICES=2 python mytrain_tf.py --model_name=unet
CUDA_VISIBLE_DEVICES=3 python mytrain_tf.py --model_name=bwunet

CUDA_VISIBLE_DEVICES=3,5,6,7 python mytrain_tf.py --data_path=/home/team19/datasets --batch_size=64 ---model_name=bwunet  --epoch=600 --model_sig=_salad

CUDA_VISIBLE_DEVICES=3,5,6,7 python mytrain_tf.py --data_path=/home/team19/datasets --batch_size=32 --model_name=bwunet  --epoch=600 --model_sig=_salad

CUDA_VISIBLE_DEVICES=7 python mytrain_tf.py --data_path=/home/team19/datasets --batch_size=8 --model_name=bwunet  --epoch=600 --model_sig=_salad