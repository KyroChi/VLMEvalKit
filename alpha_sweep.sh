#!/bin/bash
# DATASET="OCRBench POPE A-Bench_VAL ScienceQA_VAL MMBench_DEV_EN" # MMBench_QT_DEBUG_EN
DATASET="POPE MMBench_DEV_EN"
# DATASET="POPE"
BASE_OUTPUT_DIR=/home/kyle/VLMEvalKit/outputs/alpha_sweep_llava_4_bilinear
MODEL_BASE=llava_v1.5_7b
MODEL_QTVIT=llava_v1.5_7b_qtvit
# MODEL=llava_v1.5_7b_qtvit #llava_next_vicuna_7b_qtvit

# ALPHAS=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0)
ALPHAS=(0.0 1.0 2.0 3.0 4.0 5.0 6.0)

# Initialize an empty list to store the times
TIMES=()

# OUTPUT_DIR=${BASE_OUTPUT_DIR}/alpha_baseline
# mkdir -p ${OUTPUT_DIR}

# START_TIME=$(date +%s)
# python run.py \
#     --model ${MODEL_BASE} \
#     --data ${DATASET} \
#     --work-dir ${OUTPUT_DIR}

# END_TIME=$(date +%s)
# DURATION=$((END_TIME - START_TIME))
# TIMES+=("baseline,${DURATION}")
# echo "Baseline completed in ${DURATION} seconds."

for ALPHA in ${ALPHAS[@]}; do
    OUTPUT_DIR=${BASE_OUTPUT_DIR}/alpha_${ALPHA}
    mkdir -p ${OUTPUT_DIR}
    
    # Start timing
    START_TIME=$(date +%s)
    
    python run.py \
        --model ${MODEL_QTVIT} \
        --data ${DATASET} \
        --work-dir ${OUTPUT_DIR} \
        --alpha ${ALPHA}
    
    # End timing
    END_TIME=$(date +%s)
    
    # Calculate the duration
    DURATION=$((END_TIME - START_TIME))
    
    # Store the alpha and duration in the list
    TIMES+=("${ALPHA},${DURATION}")
    
    echo "Alpha ${ALPHA} completed in ${DURATION} seconds."
done

# Print the times as a table
echo -e "\nAlpha\tDuration (seconds)"
for TIME in "${TIMES[@]}"; do
    IFS=',' read -r ALPHA DURATION <<< "${TIME}"
    echo -e "${ALPHA}\t${DURATION}"
done