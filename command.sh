CUDA_VISIBLE_DEVICES=0 python mytrain_tf.py --model_name=resnet_ed
CUDA_VISIBLE_DEVICES=1 python mytrain_tf.py --model_name=resnet_flat
CUDA_VISIBLE_DEVICES=2 python mytrain_tf.py --model_name=unet
CUDA_VISIBLE_DEVICES=3 python mytrain_tf.py --model_name=bwunet

