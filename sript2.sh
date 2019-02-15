python save_CurveModel.py --dir=points2plane/difpoint565758 --init_start='curves/curve56/checkpoint-100.pt' --init_middle='curves/curve57/checkpoint-100.pt' --init_end='curves/curve58/checkpoint-100.pt'

python save_CurveModel.py --dir=points2plane/difpoint596061 --init_start='curves/curve59/checkpoint-100.pt' --init_middle='curves/curve60/checkpoint-100.pt' --init_end='curves/curve61/checkpoint-100.pt'

python save_CurveModel.py --dir=points2plane/difpoint626364 --init_start='curves/curve62/checkpoint-100.pt' --init_middle='curves/curve63/checkpoint-100.pt' --init_end='curves/curve64/checkpoint-100.pt'

python plane.py --dir=plots/difpoint565758 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/difpoint565758/checkpoint--1.pt

python plane.py --dir=plots/difpoint596061 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/difpoint596061/checkpoint--1.pt

python plane.py --dir=plots/difpoint626364  --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/difpoint626364/checkpoint--1.pt

# python train.py --dir=points2plane/connect-normal-5354 --model=VGG16 --data_path=data --epochs=100 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curves/curve53/checkpoint-100.pt --init_end=curves/curve54/checkpoint-100.pt

# python train.py --dir=points2plane/connect-normal-5556 --model=VGG16 --data_path=data --epochs=100 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curves/curve55/checkpoint-100.pt --init_end=curves/curve56/checkpoint-100.pt

# python train.py --dir=points2plane/connect-normal-5758 --model=VGG16 --data_path=data --epochs=100 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curves/curve57/checkpoint-100.pt --init_end=curves/curve58/checkpoint-100.pt

# python train.py --dir=points2plane/connect-normal-5960 --model=VGG16 --data_path=data --epochs=100 --curve=PolyChain  --num_bends=3 --fix_start --fix_end --init_start=curves/curve59/checkpoint-100.pt --init_end=curves/curve60/checkpoint-100.pt

# python plane.py --dir=plots/plot-normal-5354 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/connect-normal-5354/checkpoint-100.pt

# python plane.py --dir=plots/plot-normal-5556 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/connect-normal-5556/checkpoint-100.pt

# python plane.py --dir=plots/plot-normal-5758 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/connect-normal-5758/checkpoint-100.pt

# python plane.py --dir=plots/plot-normal-5960 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/connect-normal-5758/checkpoint-100.pt
