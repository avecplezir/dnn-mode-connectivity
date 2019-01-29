python train.py --dir=points2plane/connect_ResConv --model=VGG16 --data_path=data --epochs=100 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curves/curve/checkpoint-100.pt --init_end=curves/curve/checkpoint_rConve17-18-100.pt

python plane.py --dir=plots/plot_connect_ResConv--data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/connect_ResConv/checkpoint-100.pt

python plane_plot.py --dir=plots/plot_connect_ResConv
