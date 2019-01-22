#For training the curve:
python train.py --dir=connect_rn-normal-100 --model=VGG16 --data_path=data --epochs=100 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curve/checkpoint-100.pt --init_end=curve/checkpoint_n-100.pt

#For evaluation found checkpoint:
python eval_curve.py --dir=eval_connect_rn-normal-100 --model=VGG16 --data_path=data --curve=PolyChain  --ckpt=connect_rn-normal-100/checkpoint-100.pt

#Computing for Visualization
python plane.py --dir=plot_plane_rl-normal-100 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=connect_rn-normal-100/checkpoint-100.pt

python plane_plot.py --dir=plot_plane_rl-normal-100
