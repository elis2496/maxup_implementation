# ResNet34
#python train.py --cutmix False --m 1 --result_path ./result/Base_exp --epochs 65

# ResNet34 + CutMix
#python train.py --cutmix True --m 1 --result_path ./result/Cutmix_exp --epochs 65

# ResNet34 + CutMix + MaxUp(m=4)
#python train.py --cutmix True --m 4 --result_path ./result/Cutmix_maxup_4_exp --epochs 65

# ResNet34 + CutMix + MaxUp(m=4) (fine-tuning)
python train.py --cutmix True --m 4 --result_path ./result/Cutmix_exp_finetune --pretrained_weights ./result/Base_exp/weights.pth --epochs 65

# ResNet34 + CutMix + MaxUp(m=4) (fine-tuning)
#python train.py --cutmix True --m 20 --result_path ./result/Cutmix_maxup_20_exp_finetune --pretrained_weights ./result/Base_exp/weights.pth --epochs 65
