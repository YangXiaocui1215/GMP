for seed in 13 21 42 87 100
do
    for sl in  '8e-5' 
    do
        for num_image_tokens in 4
        do
        CUDA_VISIBLE_DEVICES=3 python twitter_sc_training_for_generated_prompt.py \
                --dataset twitter15 /home/xiaocui/code/FW-MABSA/VLP-MABSA/src/data/jsons/few_shot/twitter_2015/twitter15_${seed}_info.json \
                --checkpoint_dir ./ \
                --model_config ./config/pretrain_base.json \
                --log_dir log_for_generated_prompt/15_${seed}_sc \
                --num_beams 4 \
                --eval_every 1 \
                --lr ${sl} \
                --batch_size 4 \
                --epochs 100 \
                --grad_clip 5 \
                --warmup 0.1 \
                --is_sample 0 \
                --seed ${seed} \
                --num_image_tokens ${num_image_tokens} \
                --task twitter_sc \
                --num_workers 8 \
                --has_prompt \
                --use_caption \
                --use_generated_prompt \
                --use_different_senti_prompt
        done
    done
done
