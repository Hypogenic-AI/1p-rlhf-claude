# Downloaded Papers

## Foundational RLHF

1. **InstructGPT** (`2203.02155_instructgpt_training_lm_follow_instructions.pdf`)
   - Authors: Ouyang et al. (OpenAI)
   - Year: 2022
   - Why relevant: Established the SFT→RM→PPO pipeline; used ~40 labelers

2. **Training a Helpful and Harmless Assistant** (`2204.05862_training_helpful_harmless_assistant.pdf`)
   - Authors: Bai et al. (Anthropic)
   - Year: 2022
   - Why relevant: Created HH-RLHF dataset; identified helpfulness-harmlessness tension

3. **Learning to Summarize from Human Feedback** (`2009.01325_learning_summarize_human_feedback.pdf`)
   - Authors: Stiennon et al. (OpenAI)
   - Year: 2020
   - Why relevant: Early evidence of length bias in RLHF (1/3 of quality gap)

4. **DPO** (`2305.18290_dpo_direct_preference_optimization.pdf`)
   - Authors: Rafailov et al. (Stanford)
   - Year: 2023
   - Why relevant: RLHF without RL; shows problems persist without PPO

5. **Constitutional AI** (`2212.08073_constitutional_ai.pdf`)
   - Authors: Bai et al. (Anthropic)
   - Year: 2022
   - Why relevant: AI feedback alternative to human feedback

6. **RLAIF** (`2309.00267_rlaif_scaling_rlhf_ai_feedback.pdf`)
   - Authors: Lee et al. (Google)
   - Year: 2023
   - Why relevant: AI feedback has different verbosity bias profile

## Length Bias and Verbosity

7. **Loose Lips Sink Ships** (`2310.05199_loose_lips_length_bias_rlhf.pdf`)
   - Authors: Shen et al. (Fudan)
   - Year: 2023
   - Why relevant: Length bias originates in RM, not pretraining/SFT

8. **Verbosity Bias in Preference Labeling** (`2310.10076v1_verbosity_bias_preference_labeling.pdf`)
   - Authors: Saito et al.
   - Year: 2023
   - Why relevant: Quantified verbosity bias in both LLM and human judges

9. **Bias Fitting for Length Bias** (`2505.12843_bias_fitting_length_bias_reward_model.pdf`)
   - Year: 2025
   - Why relevant: Alternative length bias mitigation

10. **Causal Length Bias Mitigation** (`2511.12573_mitigating_length_bias_causal_lens.pdf`)
    - Year: 2025
    - Why relevant: Causal framework for length bias

## Sycophancy

11. **Towards Understanding Sycophancy** (`2310.13548v4_towards_understanding_sycophancy_v4.pdf`)
    - Authors: Sharma et al. (Anthropic/NYU)
    - Year: 2024 (ICLR)
    - Why relevant: Multi-stage sycophancy analysis; sycophancy present before RL

## Reward Model Overoptimization

12. **Scaling Laws for RM Overoptimization** (`2210.10760_scaling_laws_reward_overoptimization.pdf`)
    - Authors: Gao et al. (OpenAI)
    - Year: 2023
    - Why relevant: RM quality degrades under optimization pressure

13. **Constrained RLHF** (`2310.04373_confronting_reward_overoptimization_constrained_rlhf.pdf`)
    - Authors: Moskovitz et al.
    - Year: 2023
    - Why relevant: Multi-faceted RM constraints

14. **RM Ensembles** (`2310.02743_reward_model_ensembles_overoptimization.pdf`)
    - Authors: Coste et al.
    - Year: 2023
    - Why relevant: Ensembling reduces overoptimization

15. **Iterated RLHF Overoptimization** (`2505.18126_reward_model_overoptimization_iterated_rlhf.pdf`)
    - Year: 2025

16. **Rethinking RM Evaluation** (`2505.12763_rethinking_reward_model_evaluation_overoptimization.pdf`)
    - Year: 2025

17. **Reward Shaping** (`2502.18770_reward_shaping_mitigate_hacking.pdf`)
    - Year: 2025

## Personalized RLHF

18. **MaxMin-RLHF** (`2402.08925_maxmin_rlhf_diverse_preferences.pdf`)
    - Authors: Chakraborty et al.
    - Year: 2024
    - Why relevant: Single RM insufficient for diverse preferences

19. **Heterogeneous RLHF** (`2405.00254_rlhf_heterogeneous_feedback_personalization.pdf`)
    - Authors: Park et al.
    - Year: 2024
    - Why relevant: Formal personalization vs. aggregation framework

20. **Preference Collapse** (`2405.16455_algorithmic_bias_rlhf_preference_collapse.pdf`)
    - Authors: Li et al.
    - Year: 2024
    - Why relevant: KL regularization causes preference mode collapse

21. **Shared LoRA Personalized RLHF** (`2503.19201_shared_lora_personalized_rlhf.pdf`)
    - Year: 2025
    - Why relevant: Per-user reward models via LoRA

22. **Survey: Personalized Alignment** (`2504.07070_survey_personalized_pluralistic_alignment.pdf`)
    - Authors: Xie et al.
    - Year: 2025
    - Why relevant: Comprehensive taxonomy; verbosity is a preference dimension

23. **Swap-Guided Personalized RLHF** (`2603.12595_swap_guided_personalized_rlhf.pdf`)
    - Year: 2026 (ICLR)
    - Why relevant: Posterior collapse in personalized methods

24. **Directional Preference Alignment** (`2402.18571_directional_preference_alignment_multi_objective.pdf`)
    - Authors: Li et al.
    - Year: 2024
    - Why relevant: Multi-objective per-user preference control

25. **Interpretable Preferences** (`2406.12845_interpretable_preferences_multi_objective_reward.pdf`)
    - Authors: Wang et al.
    - Year: 2024
    - Why relevant: Decomposing preferences into interpretable dimensions
