# GMP
## Generated Multimodal Prompt

This paper, "Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt", is accepted by the Findings of ACL 2023.
The code will come soon.


## Data

The image-text data that is used in our paper can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1NgPH5xhz5dF-Zwxe-8CjjsgQJ5VaQ8KL?usp=sharing).

## Model: Generative Multimodal Prompt (GMP)

<img width="825" alt="image" src="https://github.com/YangXiaocui1215/GMP/assets/48118336/5ac903a9-26f1-4a9d-a664-e11dd73a31fc">

# Task Training
To Train the JMASA, MASC, and MATE tasks on two twitter datasets, you can just run the following code. Note that you need to change all the file path in file "GMP/src/data/jsons/few_shot_for_prompt/twitter_2015/" and "GMP/src/data/jsons/few_shot_for_prompt/twitter17_info.json" to your own path.

## For the MATE task,
```
sh scripts/15MATE_pretrain_for_generated_prompt_multitasks.sh
sh scripts/17MATE_pretrain_for_generated_prompt_multitasks.sh
```
## For the JMASA task,
```
sh scripts/15_pretrain_full_for_generated_dual_prompts_multitasks_Aspect.sh
sh scripts/17_pretrain_full_for_generated_dual_prompts_multitasks_Aspect.sh
```

# Acknowledgements
Some codes are based on the codes of [VLP-MABSA](https://github.com/NUSTM/VLP-MABSA), many thanks!
