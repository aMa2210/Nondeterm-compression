"""Convert-only puzzletron run: HF -> DeciLM teacher, activation scoring,
and uniform-width pruned child checkpoints (pipeline stages 2-4).

Skips the NAS search / MIP / realize stages entirely — the per-width child
checkpoints under {puzzle_dir}/ckpts/ffn_{K}_attn_no_op/ ARE the uniform
models we want (every layer pruned to K by modelopt's channel importance).

Run (single GPU):
    torchrun --nproc_per_node 1 modelopt_prune/run_prune_ckpts.py \
        --config modelopt_prune/config/llama-3_1-8B_pruneffn_memory.yaml
"""

import argparse
from datetime import timedelta
from pathlib import Path

import modelopt.torch.nas as mtn
import modelopt.torch.puzzletron as mtpz
import modelopt.torch.utils.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    mtpz.tools.register_hydra_resolvers()
    cfg_path = Path(args.config).resolve()
    hydra_cfg = mtpz.tools.initialize_hydra_config_for_dir(
        config_dir=str(cfg_path.parent), config_name=cfg_path.stem, overrides=[])

    timeout = getattr(hydra_cfg, "nccl_timeout_minutes", timedelta(minutes=10))
    dist.setup(timeout=timeout)

    input_model = mtpz.puzzletron_nas_plugin.PuzzletronModel()
    mtn.convert(
        input_model,
        mode=[(
            "puzzletron",
            {
                "puzzle_dir": str(hydra_cfg.puzzle_dir),
                "input_model_path": hydra_cfg.input_hf_model_path,
                "hydra_config_dir": str(cfg_path.parent),
                "hydra_config_name": cfg_path.stem,
                "dataset_path": str(hydra_cfg.dataset_path),
            },
        )],
    )
    dist.cleanup()
    mtpz.tools.mprint("[done] pruned checkpoints written under "
                      f"{hydra_cfg.puzzle_dir}/ckpts/")


if __name__ == "__main__":
    main()
