print_config "run-with-restore"
RANK_SHIFT_ON_FAILURE=$(( SLURM_NTASKS / SLURM_JOB_NUM_NODES ))
EXEC="${EXEC:-ft-restore-raxml}"

for REPEAT in ${REPEATS[@]}; do
    RBA_FILE="../rba/$DATASET.rba.$REPEAT"
    MODEL_FILE="../$DATASET.model"

    # --model "$MODEL_FILE" \ The model is part of the RBA file
    mpiexec -n $SLURM_NTASKS "../$EXEC" \
        --search \
        --msa "$RBA_FILE" \
        --tree rand{1} \
        --threads 1 \
        --seed "$SEED" \
        --force \
        --redo \
        --fail-every "$FAIL_EVERY" \
        --max-failures "$MAX_FAILURES" \
        --rank-shift-on-failure "$RANK_SHIFT_ON_FAILURE" \
        --prefix="$PREFIX_DIR/rep$REPEAT.$DATASET.seed$SEED.$NAME" \
        1> "$PREFIX_DIR/rep$REPEAT.$DATASET.seed$SEED.$NAME.out" \
        2> "$PREFIX_DIR/rep$REPEAT.$DATASET.seed$SEED.$NAME.err"
    
    mv *.overallStats.csv "$PREFIX_DIR/rep$REPEAT.$DATASET.seed$SEED.$NAME.profiler.csv"
done
