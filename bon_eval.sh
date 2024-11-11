GPUS_PER_NODE=8

MODEL_NAME="mistral-7b-bon"
CONFIG_NAME="config/7b"

CUR_DATE="notebook"

OPTS=""

# dpo version
#OPTS+=" --load /home/test/test08/yutianyu/checkpoints/20241016-020451-math-prm-top4-beat005-5e7"

#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/20241023-120131-dpo-top8"
#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/20241030-231156-dpo-icb-top8-packingbylength"
#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/dpo-top8dedup8"
#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/dpo-top8+ultrafeedback"
#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/dpo-top8+ultrainteract"
#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/20241024-121607-dpo-top8-woshuffle"
#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/20241023-135956-dpo-icb-top8"

OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints-final/ce-8192"
OPTS+=" --ref-load /home/test/testdata/models/Meta-Llama-3.1-8B-Instruct"
# OPTS+=" --tokenizer-path /home/test/testdata/models/Meta-Llama-3.1-8B-Instruct"
#OPTS+=" --ref-tokenizer-path /home/test/testdata/models/Eurux-8x22b-nca"
OPTS+=" --beta 0.05"
OPTS+=" --type dpo"

#prm with value head version
#OPTS+=" --load /home/test/test05/lwd/openrlhf-checkpoints/20241020-010155-orm_value"
#OPTS+=" --tokenizer-path /home/test/testdata/models/Meta-Llama-3.1-8B-Instruct"
#OPTS+=" --config-load /home/test/testdata/models/Meta-Llama-3.1-8B-Instruct"
#OPTS+=" --type prm-value"

# # prm as math-shepherd version
# OPTS+=" --load /home/test/test05/ylf/openrlhf/outputs/1020-math-shepherd/2e-6/llama3.1-8b-sft-template"
# OPTS+=" --tokenizer-path /home/test/test05/ylf/openrlhf/outputs/1020-math-shepherd/2e-6/llama3.1-8b-sft-template"
# OPTS+=" --type prm-llm"
# OPTS+=" --begin-of-action-token <|reserved_special_token_0|>"
# OPTS+=" --prm-token <|reserved_special_token_0|>"

OPTS+=" --bon-dataset math" #choices: math gsm8k qa
OPTS+=" --batch-size 8"
OPTS+=" --baseline 0" # output pass@k and self-consistency@n if baseline=1
OPTS+=" --combine 0" # integrate self-consistency if combine=1
OPTS+=" --orm 0" # whether to evaluate ORM
OPTS+=" $@"


CMD="python -m torch.distributed.launch --nproc_per_node=8 bon_eval.py ${OPTS}"
echo "${CMD}"
$CMD

