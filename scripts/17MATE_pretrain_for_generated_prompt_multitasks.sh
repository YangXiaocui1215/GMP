for sl in  '6.5e-5'
do 
    for seed in 13 
    do
        for num_image_tokens in 4
        do
            for loss_lambda in 0.1
            do
                CUDA_VISIBLE_DEVICES=3 python twitter_ae_training_for_generated_prompt_multitasks.py \
                        --dataset twitter17 /home/xiaocui/code/FW-MABSA/VLP-MABSA/src/data/jsons/few_shot_for_prompt/twitter_2017/twitter17_${seed}_info.json \
                        --checkpoint_dir ./ \
                        --model_config ./config/pretrain_base.json \
                        --log_dir log_for_generated_aspect_prompt_multitasks/17_${seed}_ae \
                        --num_beams 4 \
                        --eval_every 1 \
                        --lr ${sl} \
                        --batch_size 4 \
                        --epochs 100 \
                        --grad_clip 5 \
                        --warmup 0.1 \
                        --is_sample 0 \
                        --seed ${seed} \
                        --task twitter_ae \
                        --num_workers 8 \
                        --num_image_tokens ${num_image_tokens} \
                        --loss_lambda ${loss_lambda} \
                        --use_multitasks \
                        --has_prompt \
                        --use_generated_prompt \
                        --use_different_aspect_prompt  
            done
        done
    done
done