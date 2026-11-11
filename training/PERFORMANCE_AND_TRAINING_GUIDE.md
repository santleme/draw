# Draw training: performance and quality guide

Last reviewed: 2026-11-11

This guide is about useful model quality per accelerator-hour. Filling VRAM or
keeping a TPU busy is not itself a training objective. We accept a speed change
only when the same run also preserves or improves the held-out segmentation
metrics and the exported model contract.

## Current baseline

- Model: pretrained SegFormer-B0 (`nvidia/mit-b0`), four classes, 512×512 input.
- Loss: weighted cross-entropy + multiclass Dice; optimizer: AdamW; schedule:
  warmup + cosine; early stopping monitors validation mean IoU.
- CUDA baseline: BF16 where supported, otherwise FP16 with GradScaler;
  `num_workers=4`, pinned memory, persistent workers and prefetching are already
  enabled by `training/train.py`.
- The CUDA/CPU adapter enables `torch.compile` by default on CUDA (eager on
  CPU) when the config does not set `training.compile`; `--no-compile` or
  `"compile": false` keeps eager (configs t9–t15 pin `false`). Compilation is
  wrapped with a safe eager fallback: a failure at the `torch.compile` call is
  logged and training continues eager; a first-step trace/backend failure falls
  back to eager per frame (`torch._dynamo.config.suppress_errors`). The receipt
  records `efficiency.compile_status`. `channels_last` is a supported
  experiment behind `training.channels_last` / `--channels-last` (off by
  default). Both still require benchmarking, not assumed wins. The TorchTPU
  body has a separate compile path; its
  `run_t7.py` launcher gates training on a smoke, then tries compilation
  with scoped-VMEM retries and an eager fallback for E1001 failures.
- The universal launcher can select CUDA with `--backend auto` or `--backend
  cuda`; on CUDA, `--batch-size auto` passes the real forward/loss/backward
  probe to choose a batch from available VRAM.

## What has actually been measured

The user-provided TPU progress capture showed 125 training steps per epoch at
batch 32. At step 125, the epoch timer was about 244 seconds, or about 1.95 s
per batch and 16 samples/s before validation/export overhead. This is a single
observed run, not an A100 or v5 benchmark. A 30-epoch projection from that run
is about 2.0 hours of training, and roughly 2.5–3 hours including validation,
startup and export. It is an estimate, not a guarantee; use the first complete
epoch from the next run to replace it.

No A100 benchmark has been run from this workspace. The Kaggle CLI is
authenticated, but the existing `guzzbr/draw-seg-train` kernel is an old T4
MIDV document-training script with no attached datasets; it is not the Draw
SegFormer run and should not be reused as-is. The recent `draw-t7-tpu-train`
kernel is configured for Kaggle TPU v5e-8 and its last run was cancelled.

## Dataset gate before a long run

The Kaggle account currently has these relevant assets:

| Dataset | Contents/status | Training decision |
| --- | --- | --- |
| `guzzbr/draw-ink-pairs-v1` | 16,000 synthetic before/after pairs (14,000 train, 2,000 eval); structural mask checks pass, but median IoU of the predicted class-2 region against rendered darkening is 0.454 | Strong controlled training source; synthetic eval only, not a real-world quality claim |
| `guzzbr/draw-segformer-expert-pilot-v1` | 19 reviewed masks from one session per the repo record; no approved holdout | Useful for a short fine-tune/probe, not a defensible full benchmark |
| `guzzbr/draw-ink-real-v1` | 67 real image/mask pairs; labels are derived from the existing accumulator heuristic, with the hand ignored | Weak supervision/proposal source until independently corrected; not independent ground truth |
| `guzzbr/draw-real-seg-pilot-v1` | 719 frames, 707 unverified pseudo labels and 12 all-unknown, all in train across two sessions; 297 exact source frames overlap the capture pool | Exploratory only; keep overlap grouped and do not treat thumbnail triage as pixel-accurate labels |
| `guzzbr/draw-real-capture-pool-v1` | 5,086 frames across eight sessions; 4,789 unlabeled, 286 pseudo-supervised, 11 all-unknown; all `review_only`, zero human ground truth and zero training-approved. All 297 pseudo-mask proposals are from one session previously used for training | Candidate/review pool only; no independent holdout. Select whole new sessions for reviewed evaluation |
| `guzzbr/draw-ink-synth-v1` | Evaluation images without paired training masks | Evaluation/diagnostic source only |

The private Kaggle inventory and review calibration are documented in
[`docs/REVISAO_DADOS_KAGGLE_2026-11-06.md`](../docs/REVISAO_DADOS_KAGGLE_2026-11-06.md).
The 16-frame contact sheet samples only the one session with pseudo-mask
proposals and is a reviewer-calibration batch, not a holdout. The 19-frame
expert pilot is also one session; its training loss cannot support a quality
claim. The next data gate is to create an independently reviewed set from
whole capture sessions that are excluded from all training and pseudo-label
development. Continue short controlled fine-tunes on paired/synthetic data;
do not run a long model sweep or promote a candidate on synthetic metrics
alone.

## Recommended benchmark ladder

Run these stages on the exact accelerator and exact dataset intended for the
long run. Keep the output directory unique per experiment and save the config,
commit hash, accelerator name, precision and dataset manifest with every
result.

1. **Preflight:** `python training/run.py --backend auto --dry-run` and inspect
   the selected backend, then run `--smoke` to verify dependencies, model load,
   loss, backward and output paths.
2. **Short throughput sweep:** warm up first, then measure 10–20 steady-state
   training steps at a small batch/workers grid. Record samples/s, data-wait
   time, peak accelerator memory and first-step/compile time separately. Do not
   compare a compiled first step with eager steady-state steps.
3. **One complete epoch:** include validation and checkpointing. Confirm loss
   is finite and class IoU/Dice are plausible. Project total duration from this
   epoch, not from peak TFLOPs or allocated RAM.
4. **Full run:** start 30 epochs only if the first epoch's measured projection
   fits the platform runtime and the validation set is independent. Checkpoint
   each epoch and preserve the best validation checkpoint.
5. **Export gate:** evaluate the saved best checkpoint on test exactly once,
   then validate ONNX parity and the fixed visual scenarios. Do not tune against
   the test split.

Recommended initial CUDA command once the real dataset path is attached:

```bash
python training/run.py \
  --backend auto \
  --config training/configs/base.json \
  --data-dir /kaggle/input/<approved-dataset> \
  --out-dir /kaggle/working/draw-seg-bf16 \
  --epochs 30 \
  --batch-size auto \
  --precision bf16
```

`--data-dir` must point to a directory containing `train/`, `val/` and `test/`
with the paired image/mask layout described in `training/configs/README.md`;
adjust it if the Kaggle dataset has an extra wrapping directory. This command's
`--config` is forwarded only on CUDA/CPU; TPU adapters read
`training/config.json` with their own T7 overrides and generate data in-session.
Kaggle's documented run ceilings are 12 hours for GPU and 9 hours for TPU
notebooks, so save the first-epoch projection and checkpoints to persistent output.

Use Kaggle's actual runtime device printout to confirm which GPU was assigned;
requesting an A100 in metadata does not guarantee that the account/runtime has
access to it. Save logs and outputs under `/kaggle/working` so the notebook run
can preserve them.

## Hardware-specific tuning

### CUDA GPUs (A100, L4, T4, and others)

- Keep BF16 on A100 when the runtime reports support. FP16 + GradScaler is the
  fallback. FP32 is a numerical-debug mode, not the default speed setting.
- Measure the full train step before increasing batch. Use the real-step VRAM
  probe; then test nearby batch sizes and choose by stable samples/s, memory
  headroom and validation quality. A large batch that changes convergence is
  not a free speedup.
- Current input loading already uses pinned memory, persistent workers and
  prefetching. Sweep a few worker counts (for example 2, 4, 8); more workers can
  hurt a Kaggle host with only a few CPU cores or saturate storage/CPU
  augmentation. Read the assigned CPU count from the runtime before sweeping.
- Benchmark `channels_last` with BF16 as a separate experiment because the
  model is convolution-heavy: `--channels-last` (or
  `"training": {"channels_last": true}`) converts the model weights and every
  input batch, and reverts to contiguous before ONNX export. Check output
  parity and class metrics before adopting it; measure, do not assume.
- `torch.compile` is on by default on CUDA; benchmark it against
  `--no-compile` once the eager baseline is healthy. Include compilation
  startup in total wall time (it lands in the first epoch/block) and compare
  steady-state step time (`steps_steady_s_per_step` in steps mode, later
  epochs' `epoch_s` in epoch mode). Reuse a fixed shape to
  avoid recompilation. Eager stays one flag away, and compile failures fall
  back to eager automatically (`efficiency.compile_status` in the receipt).
- For cheap hypothesis checks, use steps mode (`--mode steps --max-steps 300
  --eval-interval-steps 50 --patience 2`) before paying for a full run; see
  `docs/TREINO.md`, "Reduzindo tempo por experimento".
- The current multi-GPU fallback is `DataParallel`. If the runtime exposes
  multiple GPUs and scaling matters, compare it with `DistributedDataParallel`
  in a separate adapter; do not assume a second GPU halves time.

### TPU v5e-8 / v6e

- Use BF16 and a fixed batch/shape. For XLA-based TPU training, global batch is
  sharded across cores; test batch values compatible with the device topology
  and use profiler evidence. Google's general TPU padding guidance is not a
  command to set this small SegFormer workload to batch 1024.
- Profile step time, input stalls, MXU utilization and HBM. The Kaggle v5e-8
  kernel and Colab v6e-1 runtime are different topologies and software stacks;
  transfer throughput settings only after a short run on the target runtime.
- Use the TorchTPU launcher's smoke gate before its compiled run; direct
  `train_torchtpu.py` keeps compilation behind `--compile`. Validate the full
  forward/backward path on the exact runtime image. The earlier TorchTPU
  materialization failure and later compiler failure show why TPU compilation
  cannot be presumed healthy from a successful model load.

## Quality improvements with higher expected value than using more memory

1. **Real, independent labels:** reserve a whole new capture session for
   validation/test and manually review the small, ambiguous classes—especially
   thin ink, hand-over-ink boundaries and shadows. The current pseudo masks are
   proposals, not truth.
2. **Measure by class and scenario:** gate on paper/hand/ink IoU and Dice,
   ink precision/recall with the existing 2–3 px tolerance, and the 15 fixed
   visual scenarios. Do not optimize global pixel accuracy with a dominant
   background class.
3. **Use synthetic data as controlled coverage:** balance paper/background,
   hand occlusion, thin/thick ink, lighting, camera angle and clutter. Preserve
   source-level split separation so near-duplicate scenes cannot leak across
   train and validation.
4. **Ablate model choices one at a time:** baseline B0 versus the T7 decoder
   override, encoder frozen briefly versus fully fine-tuned, and input size only
   if thin-stroke metrics remain acceptable. Compare validation curves and
   wall-clock-to-target quality, not parameter count or GPU utilization alone.
5. **Consider boundary-aware supervision only after label review:** a boundary
   head or boundary-weighted loss may help thin ink and hand edges, but noisy
   boundaries make it counterproductive. Prototype it against reviewed masks
   and retain the existing CE+Dice baseline.

## References

- [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [PyTorch data-loading optimization](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [SegFormer paper](https://arxiv.org/abs/2105.15203)
- [Google Cloud TPU performance guide](https://docs.cloud.google.com/tpu/docs/performance-guide)
- [Google Cloud TPU v6e specifications](https://docs.cloud.google.com/tpu/docs/v6e)
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle CLI kernel reference](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
