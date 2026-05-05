"""DFlash draft model configuration for Qwen3.6-35B-A3B.

Based on z-lab/Qwen3.6-35B-A3B-DFlash dimensions.
5-layer draft (best avg speedup per paper, Section 5.4.2).
"""

DRAFT_CONFIG = {
    # Architecture
    'hidden_size': 2048,
    'num_draft_layers': 5,       # paper: 5 layers best avg speedup
    'num_attention_heads': 32,   # q_proj output 4096 = 32 × 128
    'num_key_value_heads': 4,    # k_proj output 512 = 4 × 128
    'head_dim': 128,
    'intermediate_size': 6144,   # FFN intermediate
    'n_target_features': 5,      # number of target capture layers
    'rope_theta': 10000000.0,
    'rms_norm_eps': 1e-6,

    # Block diffusion
    'block_size': 16,
    'mask_token_id': 151643,  # Qwen3 MASK token (<|vision_pad|> equivalent)

    # Training
    'loss_decay': 7,           # exp(-decay * pos), paper: 7 for block 16
    'calibration_lambda': 0.1,  # weight for calibration loss
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.04,
    'max_epochs': 6,
    'max_seq_length': 3072,
    'grad_clip': 1.0,
    'batch_size': 1,            # one sequence per batch (variable length)
    'grad_accum_steps': 8,      # effective batch size = batch_size × grad_accum
    'anchors_per_sequence': 512,

    # Target model
    'target_model_id': 'Qwen/Qwen3.6-35B-A3B',
    'capture_layer_ids': [1, 7, 14, 21, 27],
    'target_gguf_path': 'models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
    'vocab_size': 151936,  # Qwen3 tokenizer vocab size

    # Paths
    'trace_dir': 'training/traces/',
    'checkpoint_dir': 'training/checkpoints/',
    'export_path': 'models/draft/qwen36-35b-a3b-dflash-v1.safetensors',
}
