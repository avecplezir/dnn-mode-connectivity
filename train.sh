for i in $(seq 4 100);
do
    python train.py --dir=curves/curve$i --model=VGG16 --data_path=data --epochs=100
done

