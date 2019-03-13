#python train.py --dir=curves_mnist/VGG16/curve3 --model=VGG16MNIST --data_path=data --epochs=20 --dataset=MNIST
#
#python train.py --dir=curves_mnist/VGG16/curve2 --model=VGG16MNIST --data_path=data --epochs=20 --dataset=MNIST
#
#python train.py --dir=curves_mnist/Linear/curve2 --model=LinearMNIST --data_path=data --epochs=20 --dataset=MNIST
#
#python train.py --dir=curves_mnist/Linear/curve3 --model=LinearMNIST --data_path=data --epochs=20 --dataset=MNIST
#
#python save_CurveModel.py --dir=points2plane/difpoint_lin_mnist --init_start=curves_mnist/Linear/curve1/checkpoint-20.pt --init_middle=curves_mnist/Linear/curve2/checkpoint-20.pt --init_end=curves_mnist/Linear/curve3/checkpoint-20.pt --model=LinearMNIST
#
#python save_CurveModel.py --dir=points2plane/difpoint_vgg_mnist --init_start=curves_mnist/VGG16/curve1/checkpoint-20.pt --init_middle=curves_mnist/VGG16/curve2/checkpoint-20.pt --init_end=curves_mnist/VGG16/curve3/checkpoint-20.pt --model=VGG16MNIST
#
#
#python train.py --dir=points2plane/connect_lin_mnist --model=LinearMNIST --data_path=data --dataset=MNIST  --epochs=20 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curves_mnist/Linear/curve1/checkpoint-20.pt --init_end=curves_mnist/Linear/curve2/checkpoint-20.pt
#
#python train.py --dir=points2plane/connect_vgg_mnist --model=VGG16MNIST --data_path=data --dataset=MNIST  --epochs=20 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curves_mnist/VGG16/curve1/checkpoint-20.pt --init_end=curves_mnist/VGG16/curve2/checkpoint-20.pt
#
#
#python plane.py --dir=plots/difpoint_lin_mnist --data_path=data --dataset=MNIST --model=LinearMNIST --curve=PolyChain --ckpt=points2plane/difpoint_lin_mnist/checkpoint--1.pt
#
#python plane.py --dir=plots/difpoint_vgg_mnist --data_path=data --dataset=MNIST --model=VGG16MNIST --curve=PolyChain --ckpt=points2plane/difpoint_vgg_mnist/checkpoint--1.pt
#
#python plane.py --dir=plots/connect_lin_mnist --data_path=data --dataset=MNIST --model=LinearMNIST --curve=PolyChain --ckpt=points2plane/connect_lin_mnist/checkpoint-20.pt
#
#python plane.py --dir=plots/connect_vgg_mnist --data_path=data --dataset=MNIST --model=VGG16MNIST --curve=PolyChain --ckpt=points2plane/connect_vgg_mnist/checkpoint-20.pt

#python plane_plot.py --dir=plots/connect_vgg_mnist
#
#python plane_plot.py --dir=plots/difpoint_lin_mnist
#
#python plane_plot.py --dir=plots/difpoint_vgg_mnist
#
#python plane_plot.py --dir=plots/connect_lin_mnist

#python plane.py --dir=plots/swap_lin_mnist --data_path=data --dataset=MNIST --model=LinearMNIST --curve=PolyChain --ckpt=points2plane/swap_lin_mnist/checkpoint--1.pt
#
#python plane.py --dir=plots/swap_vgg_mnist --data_path=data --dataset=MNIST --model=VGG16MNIST --curve=PolyChain --ckpt=points2plane/swap_vgg_mnist/checkpoint--1.pt

#python plane_plot.py --dir=plots/swap_lin_mnist
#
#python plane_plot.py --dir=plots/swap_vgg_mnist

python change_nodes.py --dir=curves/curve50 --ckpt=curves/curve50/checkpoint-100.pt --l1=-6 --l2=-3 --model=VGG16 --name=VGG16256-6-3 --data_path=data --dataset=CIFAR10 --number_swaps=256

python change_nodes.py --dir=curves/curve50 --ckpt=curves/curve50/checkpoint-100.pt --l1=-3 --l2=-1 --model=VGG16 --name=VGG16256-3-1 --data_path=data --dataset=CIFAR10 --number_swaps=256

python save_CurveModel.py --dir=points2plane/LinSwapp256 --init_start=curves/curve50/checkpoint-100.pt --init_middle=curves/curve50/VGG16256-6-3-100.pt --init_end=curves/curve50/VGG16256-3-1-100.pt --model=VGG16

python plane.py --dir=plots/LinSwapp256 --data_path=data --dataset=CIFAR10 --model=VGG16 --curve=PolyChain --ckpt=points2plane/LinSwapp256/checkpoint--1.pt