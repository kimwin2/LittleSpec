CUDA_VISIBLE_DEVICES="0" python3 sd_eval.py \
    --model_id /home/gpu1/emtechllm/bs93.lee/LittleSpec/ckpts_lb_plus/llama2_7b/spec_llama2_7b_total1p0bit_resume0p1_SmoothSign_lr4e-5_bs4_wRes_LittleBitITQLinear/2026_02_20_00_03 \
    --quant_func $FUNC \
    --quant_mod LittleBitITQLinear \
    --eff_bit 1.0 \
    --resume_eff_bit 0.1 \
    --sd_k_list "2,4,6,8" \
    --sd_num_prompt 50 \
    --zeroshot_task "piqa" \
    --model_type llama \
    --run_sd_offline True \
    --run_sd_online True \




