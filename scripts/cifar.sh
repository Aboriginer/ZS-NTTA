testsets=('CIFAR-10' 'CIFAR-100')
gpu=0
OODs=('SVHN' 'LSUN' 'Texture' 'Places')

# # ZS-CLIP
# methods=('zs_clip_configs.py')
# # SoTTA
# methods=('sotta_configs.py')
# # TDA
# methods=('tda_configs.py')
# # Tent
# methods=('tent_configs.py')
# # TPT
# methods=('tpt_configs.py')
# # AdaND (Ours)
methods=('zs_noisytta_configs.py')

for testset in "${testsets[@]}"; do
    for OOD in "${OODs[@]}"; do
        for method in "${methods[@]}"; do
            echo "Running experiment: ID=${testset}, OOD=${OOD}"
            python ./main.py \
                --config configs/$method \
                --test_set $testset \
                --OOD_set $OOD \
                --gpu $gpu 
        done
    done
done
