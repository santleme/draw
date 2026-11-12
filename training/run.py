"""Universal dispatcher for the Draw SegFormer training adapters.

The selected Kaggle/Colab image owns the accelerator framework. This script
detects the backend without importing torch in a TPU parent process, then
starts the matching existing adapter in a fresh process:

* CUDA GPU (A100, T4, L4, H100, etc.) -> train.py
* Kaggle TPU / torch_xla -> train_tpu.py
* Colab TPU / torch_tpu -> run_t7.py (probe + smoke gate + training)

Use ``python training/run.py --backend auto --dry-run`` to inspect routing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def _batch_size(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use 'auto' ou um inteiro positivo") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("batch size precisa ser positivo")
    return parsed


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _has_tpu_device() -> bool:
    if any(
        os.environ.get(key)
        for key in (
            "TPU_NAME",
            "TPU_PROCESS_ADDRESSES",
            "CLOUD_TPU_TASK_ID",
            "COLAB_TPU_ADDR",
        )
    ):
        return True
    return any(Path("/dev").glob("accel*"))


def detect_backend(requested: str) -> str:
    if requested != "auto":
        return requested

    # torch_tpu registers and initializes the native backend during import.
    # Detect it by package + accelerator presence; do not import it here.
    if _has_tpu_device() and _has_module("torch_tpu"):
        return "torchtpu"
    if _has_tpu_device() and _has_module("torch_xla"):
        return "xla"

    # On CUDA runtimes it is safe to import the runtime-provided torch. It is
    # intentionally imported only after TPU detection to avoid initializing
    # PJRT in the dispatcher process.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch não está instalado nesta sessão. Selecione um runtime "
            "GPU/TPU com o framework fornecido pela plataforma."
        ) from exc
    if torch.cuda.is_available():
        return "cuda"

    # Some TPU images do not expose /dev/accel to the Python container. The
    # explicit package fallback keeps auto useful there; adapter preflight
    # will still fail clearly if that image has no live TPU.
    if _has_module("torch_tpu"):
        return "torchtpu"
    if _has_module("torch_xla"):
        return "xla"
    return "cpu"


def build_command(args: argparse.Namespace, backend: str) -> list[str]:
    py = sys.executable
    data = str(Path(args.data_dir).expanduser())
    out = str(Path(args.out_dir).expanduser())

    if backend in ("cuda", "cpu"):
        cmd = [
            py,
            str(HERE / "train.py"),
            "--config",
            args.config,
            "--data-dir",
            data,
            "--out-dir",
            out,
        ]
        if args.real_train_dir:
            cmd += ["--real-train-dir", args.real_train_dir]
        if args.allow_unverified_real_train:
            cmd.append("--allow-unverified-real-train")
        if args.real_loss_weight is not None:
            cmd += ["--real-loss-weight", str(args.real_loss_weight)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size == "auto":
            cmd.append("--probe-batch")
        else:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.precision is not None:
            cmd += ["--precision", args.precision]
        if getattr(args, "compile", None) is not None:
            cmd.append("--compile" if args.compile else "--no-compile")
        if getattr(args, "channels_last", None) is not None:
            cmd.append(
                "--channels-last" if args.channels_last else "--no-channels-last"
            )
        for flag in ("mode", "max_steps", "eval_interval_steps", "patience"):
            value = getattr(args, flag, None)
            if value is not None:
                cmd += ["--" + flag.replace("_", "-"), str(value)]
        if args.smoke:
            cmd += ["--smoke", "--smoke-dir", args.smoke_dir]
        return cmd

    if backend == "xla":
        if getattr(args, "recipe", "segformer") == "finetune-real":
            # Frente 54: o fine-tune de tinta em TPU vive em
            # training/finetune_tpu.py (port do finetune_real); a
            # rota SegFormer (train_tpu.py, abaixo) nao muda.
            cmd = [
                py,
                str(HERE / "finetune_tpu.py"),
                "--config",
                args.config,
                "--data-dir",
                data,
                "--out-dir",
                out,
            ]
            if args.batch_size != "auto":
                cmd += ["--batch-per-core", str(args.batch_size)]
            if args.smoke:
                cmd.append("--smoke")
            return cmd
        if args.real_train_dir:
            raise ValueError("este adaptador TPU ainda não aceita --real-train-dir")
        cmd = [py, str(HERE / "train_tpu.py"), "--data-dir", data, "--out-dir", out]
        if args.smoke:
            smoke_root = Path(args.smoke_dir)
            cmd[cmd.index("--data-dir") + 1] = str(smoke_root / "data")
            cmd[cmd.index("--out-dir") + 1] = str(smoke_root / "out")
            cmd += [
                "--epochs",
                "1",
                "--train-count",
                "96",
                "--val-count",
                "16",
                "--test-count",
                "16",
            ]
        elif args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if not args.smoke:
            for flag, value in (
                ("--train-count", args.train_count),
                ("--val-count", args.val_count),
                ("--test-count", args.test_count),
            ):
                if value is not None:
                    cmd += [flag, str(value)]
        if args.batch_size != "auto":
            cmd += ["--batch-per-core", str(args.batch_size)]
        if args.precision is not None:
            cmd += ["--precision", args.precision]
        return cmd

    if backend == "torchtpu":
        if args.real_train_dir:
            raise ValueError("este adaptador TPU ainda não aceita --real-train-dir")
        cmd = [
            py,
            str(HERE / "run_t7.py"),
            "--data-dir",
            data,
            "--out-dir",
            out,
            "--smoke-out-dir",
            args.smoke_dir,
        ]
        if not args.smoke and args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if not args.smoke:
            for flag, value in (
                ("--train-count", args.train_count),
                ("--val-count", args.val_count),
                ("--test-count", args.test_count),
            ):
                if value is not None:
                    cmd += [flag, str(value)]
            if args.budget_min is not None:
                cmd += ["--budget-min", str(args.budget_min)]
        if args.batch_size != "auto":
            cmd += ["--batch", str(args.batch_size)]
        if args.precision is not None:
            cmd += ["--precision", args.precision]
        if args.smoke:
            cmd.append("--smoke-only")
        return cmd

    raise ValueError(f"backend não suportado: {backend}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Treino Draw com detecção automática de GPU/TPU."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cuda", "xla", "torchtpu", "cpu"),
        default="auto",
        help="auto detecta o runtime; use um valor para fixá-lo.",
    )
    parser.add_argument(
        "--recipe",
        choices=("segformer", "finetune-real"),
        default="segformer",
        help="segformer: training/train.py ou train_tpu.py (o "
        "comportamento de sempre); finetune-real: o fine-tune de "
        "tinta training/finetune_real.py em CUDA ou "
        "training/finetune_tpu.py em TPU (frente 54).",
    )
    parser.add_argument("--config", default="training/configs/base.json")
    parser.add_argument("--data-dir", default="dataset/synthetic")
    parser.add_argument("--real-train-dir", default=None)
    parser.add_argument("--allow-unverified-real-train", action="store_true")
    parser.add_argument(
        "--real-loss-weight",
        type=float,
        default=None,
        help="Peso por amostra do pacote --real-train-dir "
        "na loss (0, 1]. Ausente = o da config "
        "(loss.unverified_real_sample_weight). Use 1.0 "
        "para pacotes com ground truth verificado "
        "(ex.: SmartDoc convertido de quads oficiais).",
    )
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--budget-min",
        type=float,
        default=None,
        help="Só rota TorchTPU (Colab): teto de "
        "parede em minutos; o corpo para no "
        "fim de uma época e entrega o melhor "
        "checkpoint (ausente = sem teto).",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Só rotas TPU (dados gerados em sessão): "
        "imagens de treino (padrão do corpo: 4000).",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Só rotas TPU: imagens de validação (400).",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Só rotas TPU: imagens de teste (400).",
    )
    parser.add_argument(
        "--batch-size",
        type=_batch_size,
        default="auto",
        help="auto usa a sonda de VRAM em CUDA; em TPU mantém o lote padrão.",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default=None,
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=None,
        help="Só rota CUDA/CPU: liga torch.compile no treino.",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Só rota CUDA/CPU: desliga torch.compile.",
    )
    parser.add_argument(
        "--channels-last",
        dest="channels_last",
        action="store_true",
        default=None,
        help="Só rota CUDA/CPU: experimento channels_last (meça antes de adotar).",
    )
    parser.add_argument(
        "--no-channels-last",
        dest="channels_last",
        action="store_false",
        help="Só rota CUDA/CPU: desliga channels_last.",
    )
    parser.add_argument(
        "--mode",
        choices=("epochs", "steps"),
        default=None,
        help="Só rota CUDA/CPU: training.mode (steps = iteração rápida).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Só rota CUDA/CPU: training.max_steps do modo steps.",
    )
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=None,
        help="Só rota CUDA/CPU: passos por bloco de avaliação no modo steps.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Só rota CUDA/CPU: early_stopping.patience_epochs (épocas ou blocos).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Executa a etapa curta/prova do adaptador selecionado.",
    )
    parser.add_argument("--smoke-dir", default="/tmp/draw-smoke")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o backend e o comando sem iniciar o treino.",
    )
    args = parser.parse_args(argv)

    try:
        backend = detect_backend(args.backend)
    except RuntimeError as exc:
        parser.error(str(exc))
    if backend != "cuda" and args.precision == "fp16":
        parser.error("fp16 está disponível só em CUDA; use bf16 ou fp32 em TPU")
    try:
        command = build_command(args, backend)
    except ValueError as exc:
        parser.error(str(exc))

    if backend == "cuda":
        if args.dry_run:
            hardware = "BACKEND cuda | hardware ainda não consultado (dry-run)"
        else:
            import torch

            if not torch.cuda.is_available():
                parser.error(
                    "backend CUDA selecionado, mas torch.cuda.is_available() é falso"
                )
            device = torch.cuda.get_device_properties(0)
            hardware = (
                f"BACKEND cuda | {torch.cuda.get_device_name(0)} | "
                f"GPUs={torch.cuda.device_count()} | "
                f"VRAM={device.total_memory / 2**30:.1f} GiB"
            )
        print(hardware, flush=True)
    else:
        print(f"BACKEND {backend}", flush=True)
    print("COMMAND", " ".join(command), flush=True)
    if args.dry_run:
        print("DRY_RUN_OK", flush=True)
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
