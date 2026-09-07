from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import evaluate_small_subset
from .inference import (
    apply_optional_trigger,
    load_image_chw_01,
    load_model_from_checkpoint,
    predict_one_image,
)
from .paths import ensure_dir, find_checkpoint, load_final_summary, save_json, stage_display_name, stage_output_name
from .subset import count_records, list_image_records, pick_first_non_target_image
from .visualization import plot_demo_training_curve


def _stage_prefix(trigger_type: str) -> tuple[str, str]:
    if str(trigger_type).lower().strip() == "square":
        return "Square Baseline", "Square"
    return "Checkerboard Improved", "Checkerboard"


def _load_demo_training_arrays(
    subset_train_dir: Path,
    trigger_type: str,
    target_label: int,
    trigger_size: int,
    alpha: float,
    position: str,
) -> tuple[np.ndarray, np.ndarray]:
    records = list_image_records(subset_train_dir)
    if not records:
        raise ValueError(f"Demo train subset is empty: {subset_train_dir}")

    clean_images = np.stack([load_image_chw_01(path) for path, _ in records]).astype(np.float32)
    clean_labels = np.asarray([label for _, label in records], dtype=np.int32)

    poisoned_images: list[np.ndarray] = []
    poisoned_labels: list[int] = []
    for image, label in zip(clean_images, clean_labels):
        if int(label) == int(target_label):
            continue
        poisoned_images.append(
            apply_optional_trigger(
                image,
                trigger_type=trigger_type,
                trigger_size=trigger_size,
                alpha=alpha,
                position=position,
            )
        )
        poisoned_labels.append(int(target_label))

    if poisoned_images:
        train_images = np.concatenate([clean_images, np.stack(poisoned_images).astype(np.float32)], axis=0)
        train_labels = np.concatenate([clean_labels, np.asarray(poisoned_labels, dtype=np.int32)], axis=0)
    else:
        train_images = clean_images
        train_labels = clean_labels
    return train_images.astype(np.float32), train_labels.astype(np.int32)


def _normalize_for_model(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) * 2.0 - 1.0


def _save_checkpoint(model, ckpt_path: Path) -> str:
    import shutil
    import mindspore as ms

    try:
        ms.save_checkpoint(model, str(ckpt_path))
        return "native"
    except Exception as exc:  # noqa: BLE001
        tmp_path = ckpt_path.with_suffix(".tmp")
        if tmp_path.exists():
            shutil.copy2(tmp_path, ckpt_path)
            return f"copied_from_tmp_after_error: {exc}"
        raise


def _save_training_curve_csv(epoch_metrics: list[dict[str, Any]], csv_path: Path) -> Path:
    fieldnames = ["epoch", "train_loss", "train_batches", "clean_accuracy", "asr", "elapsed_seconds"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in epoch_metrics:
            writer.writerow({key: item.get(key) for key in fieldnames})
    return csv_path


def _save_training_curve_png(log_path: Path, output_path: Path, label: str) -> Path:
    import matplotlib.pyplot as plt

    figure = plot_demo_training_curve(log_path, labels=[label])
    figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _resolve_stage_checkpoint(
    *,
    demo_checkpoint_path: Path,
    fallback_checkpoint_path: Path,
    prefer_live_trained_checkpoint: bool,
) -> tuple[Path, str, str | None]:
    if bool(prefer_live_trained_checkpoint) and demo_checkpoint_path.exists():
        return demo_checkpoint_path, "live_trained_demo_checkpoint", None
    if bool(prefer_live_trained_checkpoint):
        reason = f"Live demo checkpoint is missing, so fallback was used: {demo_checkpoint_path}"
        return fallback_checkpoint_path, "official_checkpoint_fallback", reason
    reason = "Live demo checkpoint preference is disabled by config."
    return fallback_checkpoint_path, "official_checkpoint_fallback", reason


def _formal_eval_summary(experiment_name: str, trigger_type: str, target_label: int) -> dict[str, Any]:
    for item in load_final_summary():
        experiment = item.get("experiment") or item.get("experiment_name")
        if experiment != experiment_name:
            continue
        clean_accuracy = item.get("clean_accuracy", item.get("clean_acc", 0.0))
        asr = item.get("asr", item.get("attack_success_rate", 0.0))
        avg_target_confidence = item.get("avg_target_confidence_on_triggered", item.get("avg_target_confidence", 0.0))
        return {
            "trigger_type": str(trigger_type),
            "clean_accuracy": float(clean_accuracy),
            "attack_success_rate": float(asr),
            "asr": float(asr),
            "target_label": int(target_label),
            "avg_target_confidence_on_triggered": float(avg_target_confidence or 0.0),
            "num_clean_eval_samples": int(item.get("num_clean_eval_samples", item.get("clean_samples", 0)) or 0),
            "num_triggered_eval_samples": int(item.get("num_triggered_eval_samples", item.get("triggered_samples", 0)) or 0),
            "checkpoint_path": str(item.get("checkpoint", item.get("ckpt_path", item.get("checkpoint_path", "")))),
            "checkpoint_loaded": True,
            "metric_source": "formal_server_summary",
        }
    raise ValueError(f"final_summary 中没有找到实验结果：{experiment_name}")


def run_small_demo_training(
    stage_name: str,
    trigger_type: str,
    checkpoint_path: str | Path | None,
    subset_train_dir: str | Path,
    subset_test_dir: str | Path,
    output_dir: str | Path,
    target_label: int = 0,
    epochs: int = 1,
    batch_size: int = 16,
    device_target: str = "auto",
    ms_mode: str = "GRAPH",
    num_classes: int = 43,
    norm_type: str = "group",
    trigger_size: int = 4,
    alpha: float = 0.8,
    position: str = "bottom_right",
    seed: int = 42,
    max_steps_per_epoch: int | None = None,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    experiment_name: str | None = None,
    eval_each_epoch: bool = True,
    save_epoch_metrics: bool = True,
    save_training_curve: bool = True,
    train_scope: str = "classifier",
    run_live_eval: bool = False,
) -> dict[str, Any]:
    """Run a tiny real training loop for live demonstration only."""

    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops

    class StableSparseCrossEntropy(nn.Cell):
        def __init__(self) -> None:
            super().__init__()
            self.cast = ops.Cast()
            self.expand_dims = ops.ExpandDims()
            self.reduce_max = ops.ReduceMax(keep_dims=True)
            self.reduce_sum = ops.ReduceSum(keep_dims=True)
            self.reduce_mean = ops.ReduceMean(keep_dims=False)
            self.gather_d = ops.GatherD()
            self.reshape = ops.Reshape()

        def construct(self, logits, labels):
            logits = self.cast(logits, ms.float32)
            labels = self.cast(labels, ms.int32)
            label_index = self.expand_dims(labels, 1)
            max_logits = self.reduce_max(logits, 1)
            shifted_logits = logits - max_logits
            sum_exp = self.reduce_sum(ops.exp(shifted_logits), 1)
            logsumexp = ops.log(sum_exp) + max_logits
            true_label_logits = self.gather_d(logits, 1, label_index)
            per_sample_loss = self.reshape(logsumexp - true_label_logits, (-1,))
            return self.reduce_mean(per_sample_loss)

    def _resolve_trainable_parameters(model, scope: str):
        def _collect_cell_parameters(cell) -> list:
            parameters = list(cell.trainable_params())
            if parameters:
                return parameters
            get_parameters = getattr(cell, "get_parameters", None)
            if callable(get_parameters):
                try:
                    parameters = list(get_parameters(expand=True))
                except TypeError:
                    parameters = list(get_parameters())
                if parameters:
                    return parameters
            parameters_and_names = getattr(cell, "parameters_and_names", None)
            if callable(parameters_and_names):
                return [parameter for _, parameter in parameters_and_names()]
            return []

        normalized_scope = str(scope).strip().lower()
        if normalized_scope in {"classifier", "head", "linear"}:
            classifier = getattr(model, "classifier", None)
            if classifier is None:
                raise AttributeError("当前模型没有 classifier，无法只微调分类头。")
            classifier_parameters = _collect_cell_parameters(classifier)
            if not classifier_parameters:
                raise ValueError("classifier 参数列表为空，无法只微调分类头。")
            for parameter in model.trainable_params():
                try:
                    parameter.requires_grad = False
                except Exception:
                    pass
            for parameter in classifier_parameters:
                try:
                    parameter.requires_grad = True
                except Exception:
                    pass
            return classifier_parameters, "classifier"
        if normalized_scope in {"all", "full"}:
            for parameter in model.trainable_params():
                try:
                    parameter.requires_grad = True
                except Exception:
                    pass
            all_parameters = list(model.trainable_params())
            if not all_parameters:
                all_parameters = _collect_cell_parameters(model)
            return all_parameters, "all"
        raise ValueError("train_scope 只能是 'classifier' 或 'all'。")

    def _scalar_float(value) -> float:
        if hasattr(value, "asnumpy"):
            value = value.asnumpy()
        return float(np.asarray(value, dtype=np.float32).mean())

    requested_epochs = int(epochs)
    if requested_epochs <= 0:
        raise ValueError(f"epochs must be >= 1, got {requested_epochs}")

    output_dir = ensure_dir(Path(output_dir))
    subset_train_dir = Path(subset_train_dir)
    subset_test_dir = Path(subset_test_dir)
    if checkpoint_path is None:
        if not experiment_name:
            raise ValueError("experiment_name is required when checkpoint_path is not provided.")
        ckpt_path = find_checkpoint(experiment_name)
    else:
        ckpt_path = Path(checkpoint_path)

    model = load_model_from_checkpoint(
        experiment_name=experiment_name or ckpt_path.stem,
        checkpoint_path=ckpt_path,
        num_classes=num_classes,
        norm_type=norm_type,
        device_target=device_target,
        ms_mode=ms_mode,
    )
    actual_device = str(getattr(model, "demo_device_target", device_target))
    train_images, train_labels = _load_demo_training_arrays(
        subset_train_dir=subset_train_dir,
        trigger_type=trigger_type,
        target_label=target_label,
        trigger_size=trigger_size,
        alpha=alpha,
        position=position,
    )
    rng = random.Random(int(seed))

    loss_fn = StableSparseCrossEntropy()
    trainable_parameters, resolved_train_scope = _resolve_trainable_parameters(model, train_scope)
    optimizer = nn.Momentum(
        trainable_parameters,
        learning_rate=float(learning_rate),
        momentum=0.9,
        weight_decay=1e-4,
        use_nesterov=True,
    )
    optimizer_info = {
        "name": optimizer.__class__.__name__,
        "learning_rate": float(learning_rate),
        "grad_clip": float(grad_clip),
        "grad_clip_applied": bool(float(grad_clip) > 0.0 and resolved_train_scope == "all"),
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "use_nesterov": True,
        "train_scope": resolved_train_scope,
        "trainable_parameter_count": len(trainable_parameters),
    }

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss

    value_and_grad = getattr(ms, "value_and_grad", None) or getattr(ops, "value_and_grad")
    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters)

    epoch_metrics: list[dict[str, Any]] = []
    step_loss_log: list[dict[str, Any]] = []
    epoch_label, step_label = _stage_prefix(trigger_type)
    total_steps = int(math.ceil(len(train_labels) / int(batch_size)))
    if max_steps_per_epoch is not None:
        total_steps = min(total_steps, int(max_steps_per_epoch))

    train_start = time.perf_counter()
    last_eval_summary: dict[str, Any] | None = None
    print(
        f"[{step_label} Training] optimizer={optimizer_info['name']} "
        f"lr={optimizer_info['learning_rate']} "
        f"grad_clip={optimizer_info['grad_clip']} "
        f"clip_applied={optimizer_info['grad_clip_applied']} "
        f"scope={optimizer_info['train_scope']} "
        f"trainable_params={optimizer_info['trainable_parameter_count']} "
        f"loss={loss_fn.__class__.__name__}"
    )

    for epoch in range(1, requested_epochs + 1):
        indices = list(range(len(train_labels)))
        rng.shuffle(indices)
        losses: list[float] = []
        train_batches = 0
        model.set_train(True)

        for start in range(0, len(indices), int(batch_size)):
            if max_steps_per_epoch is not None and train_batches >= int(max_steps_per_epoch):
                break
            batch_indices = indices[start : start + int(batch_size)]
            batch_x = _normalize_for_model(train_images[batch_indices])
            batch_y = train_labels[batch_indices]
            loss, grads = grad_fn(ms.Tensor(batch_x, ms.float32), ms.Tensor(batch_y, ms.int32))
            loss_value = _scalar_float(loss)
            if not np.isfinite(loss_value):
                raise RuntimeError(
                    f"{step_label} training produced non-finite loss before optimizer update: "
                    f"epoch={epoch}, step={train_batches + 1}, loss={loss_value}"
                )
            if bool(optimizer_info["grad_clip_applied"]):
                clipped_grads = ops.clip_by_global_norm(grads, float(grad_clip))
            else:
                clipped_grads = grads
            optimizer(clipped_grads)
            train_batches += 1
            losses.append(loss_value)
            step_record = {
                "epoch": epoch,
                "step": train_batches,
                "total_steps": total_steps,
                "loss": loss_value,
            }
            step_loss_log.append(step_record)
            if train_batches <= 3 or train_batches == total_steps:
                print(f"[{step_label} Step {train_batches}/{total_steps}] loss={loss_value:.6f}")

        train_loss = float(np.mean(losses)) if losses else 0.0
        epoch_eval_summary: dict[str, Any] | None = None
        if bool(eval_each_epoch) and bool(run_live_eval):
            epoch_eval_summary = evaluate_small_subset(
                model=model,
                subset_test_dir=subset_test_dir,
                trigger_type=trigger_type,
                target_label=target_label,
                trigger_size=trigger_size,
                alpha=alpha,
                position=position,
                batch_size=1,
            )
            last_eval_summary = epoch_eval_summary

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_batches": int(train_batches),
            "clean_accuracy": (
                float(epoch_eval_summary["clean_accuracy"]) if epoch_eval_summary is not None else None
            ),
            "asr": float(epoch_eval_summary["attack_success_rate"]) if epoch_eval_summary is not None else None,
            "metric_source": (
                str(epoch_eval_summary.get("metric_source", "live_demo_subset"))
                if epoch_eval_summary is not None
                else None
            ),
            "elapsed_seconds": float(time.perf_counter() - train_start),
        }
        epoch_metrics.append(epoch_record)
        if epoch_eval_summary is None:
            print(f"[{epoch_label} Epoch {epoch}/{requested_epochs}] train_loss={train_loss:.6f}")
        else:
            print(
                f"[{epoch_label} Epoch {epoch}/{requested_epochs}] "
                f"train_loss={train_loss:.6f} "
                f"clean_acc={epoch_record['clean_accuracy']:.4f} "
                f"asr={epoch_record['asr']:.4f}"
            )

    if last_eval_summary is None and bool(run_live_eval):
        last_eval_summary = evaluate_small_subset(
            model=model,
            subset_test_dir=subset_test_dir,
            trigger_type=trigger_type,
            target_label=target_label,
            trigger_size=trigger_size,
            alpha=alpha,
            position=position,
            batch_size=1,
        )
        if epoch_metrics:
            epoch_metrics[-1]["clean_accuracy"] = float(last_eval_summary["clean_accuracy"])
            epoch_metrics[-1]["asr"] = float(last_eval_summary["attack_success_rate"])
    if last_eval_summary is None:
        last_eval_summary = _formal_eval_summary(
            experiment_name=experiment_name or Path(ckpt_path).stem,
            trigger_type=trigger_type,
            target_label=target_label,
        )

    demo_checkpoint_path = output_dir / "demo_last.ckpt"
    ckpt_save_method = _save_checkpoint(model, demo_checkpoint_path)
    curve_csv_path = output_dir / "demo_training_curve.csv"
    curve_png_path = output_dir / "training_curve.png"
    train_log_path = output_dir / "demo_train_log.json"

    log = {
        "trigger_type": trigger_type,
        "stage_name": stage_name,
        "device_target": actual_device,
        "ms_mode": str(ms_mode).upper().strip(),
        "checkpoint_loaded": True,
        "checkpoint_path": str(ckpt_path),
        "subset_train_count": int(count_records(subset_train_dir)),
        "subset_test_count": int(count_records(subset_test_dir)),
        "demo_training_sample_count_after_trigger_injection": int(len(train_labels)),
        "epochs": requested_epochs,
        "epochs_completed": int(len(epoch_metrics)),
        "batch_size": int(batch_size),
        "target_label": int(target_label),
        "learning_rate": float(learning_rate),
        "grad_clip": float(grad_clip),
        "grad_clip_applied": bool(optimizer_info["grad_clip_applied"]),
        "optimizer": optimizer_info,
        "loss_name": loss_fn.__class__.__name__,
        "eval_each_epoch": bool(eval_each_epoch),
        "run_live_eval": bool(run_live_eval),
        "save_epoch_metrics": bool(save_epoch_metrics),
        "save_training_curve": bool(save_training_curve),
        "train_scope": resolved_train_scope,
        "trainable_parameter_count": len(trainable_parameters),
        "train_loss_by_epoch": [item["train_loss"] for item in epoch_metrics],
        "clean_accuracy_by_epoch": [item["clean_accuracy"] for item in epoch_metrics],
        "asr_by_epoch": [item["asr"] for item in epoch_metrics],
        "mini_clean_accuracy_by_epoch": [item["clean_accuracy"] for item in epoch_metrics],
        "mini_asr_by_epoch": [item["asr"] for item in epoch_metrics],
        "epoch_metrics": epoch_metrics,
        "epochs_history": epoch_metrics,
        "step_loss_log": step_loss_log,
        "demo_checkpoint_path": str(demo_checkpoint_path),
        "demo_checkpoint_save_method": ckpt_save_method,
        "demo_training_curve_csv_path": str(curve_csv_path),
        "training_curve_path": str(curve_png_path),
        "final_clean_accuracy": float(last_eval_summary["clean_accuracy"]),
        "final_asr": float(last_eval_summary["attack_success_rate"]),
        "official_metric_warning": (
            "Formal metrics come from the full server-side training results. "
            "The live demo subset only proves the onsite execution path."
        ),
    }
    save_json(log, train_log_path)
    _save_training_curve_csv(epoch_metrics, curve_csv_path)
    if bool(save_training_curve):
        _save_training_curve_png(train_log_path, curve_png_path, label=trigger_type)

    save_json(
        {
            "stage_name": stage_name,
            "trigger_type": trigger_type,
            "target_label": int(target_label),
            "epochs": requested_epochs,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "device_target": actual_device,
            "ms_mode": str(ms_mode).upper().strip(),
            "checkpoint_path": str(ckpt_path),
            "trigger_size": int(trigger_size),
            "alpha": float(alpha),
            "position": position,
            "subset_train_dir": str(subset_train_dir),
            "subset_test_dir": str(subset_test_dir),
            "learning_rate": float(learning_rate),
            "grad_clip": float(grad_clip),
            "grad_clip_applied": bool(optimizer_info["grad_clip_applied"]),
            "optimizer": optimizer_info,
            "loss_name": loss_fn.__class__.__name__,
            "eval_each_epoch": bool(eval_each_epoch),
            "run_live_eval": bool(run_live_eval),
            "save_epoch_metrics": bool(save_epoch_metrics),
            "save_training_curve": bool(save_training_curve),
        },
        output_dir / "demo_config_used.json",
    )
    return log


def run_stage_pipeline(
    stage_config: dict[str, Any],
    run_root: str | Path,
    subset_root: str | Path,
    target_label: int,
    epochs: int,
    batch_size: int,
    device_target: str,
    ms_mode: str,
    seed: int,
    num_classes: int = 43,
    norm_type: str = "group",
    max_steps_per_epoch: int | None = None,
    grad_clip: float = 1.0,
    eval_each_epoch: bool = True,
    save_epoch_metrics: bool = True,
    save_training_curve: bool = True,
    prefer_live_trained_checkpoint: bool = True,
    train_scope: str = "classifier",
    run_live_eval: bool = False,
) -> dict[str, Any]:
    """Run the full onsite stage with live training artifacts and live-checkpoint eval."""

    trigger_type = str(stage_config["trigger_type"])
    experiment_name = str(stage_config["experiment_name"])
    display_name = stage_display_name(trigger_type)
    output_dir = ensure_dir(Path(run_root) / stage_output_name(trigger_type))
    subset_root = Path(subset_root)
    subset_train_dir = subset_root / "train"
    subset_test_dir = subset_root / "test"
    official_checkpoint_path = find_checkpoint(experiment_name)

    print(f"Stage: {display_name}")
    print(f"Official checkpoint: {official_checkpoint_path}")
    train_log = run_small_demo_training(
        stage_name=display_name,
        trigger_type=trigger_type,
        experiment_name=experiment_name,
        checkpoint_path=official_checkpoint_path,
        subset_train_dir=subset_train_dir,
        subset_test_dir=subset_test_dir,
        output_dir=output_dir,
        target_label=target_label,
        epochs=epochs,
        batch_size=batch_size,
        device_target=device_target,
        ms_mode=ms_mode,
        num_classes=num_classes,
        norm_type=norm_type,
        trigger_size=int(stage_config.get("trigger_size", 4)),
        alpha=float(stage_config.get("alpha", 0.8)),
        position=str(stage_config.get("position", "bottom_right")),
        seed=seed,
        max_steps_per_epoch=max_steps_per_epoch,
        grad_clip=grad_clip,
        eval_each_epoch=eval_each_epoch,
        save_epoch_metrics=save_epoch_metrics,
        save_training_curve=save_training_curve,
        train_scope=train_scope,
        run_live_eval=run_live_eval,
    )

    effective_checkpoint_path, model_source, model_source_reason = _resolve_stage_checkpoint(
        demo_checkpoint_path=Path(train_log["demo_checkpoint_path"]),
        fallback_checkpoint_path=official_checkpoint_path,
        prefer_live_trained_checkpoint=prefer_live_trained_checkpoint,
    )
    clean_result: dict[str, Any] = {}
    triggered_result: dict[str, Any] = {}
    if bool(run_live_eval):
        stage_model = load_model_from_checkpoint(
            experiment_name=experiment_name,
            checkpoint_path=effective_checkpoint_path,
            num_classes=num_classes,
            norm_type=norm_type,
            device_target=device_target,
            ms_mode=ms_mode,
        )
        sample_image = pick_first_non_target_image(subset_test_dir, target_label=target_label)

        clean_result = predict_one_image(stage_model, sample_image, trigger_type=None, target_label=target_label)
        triggered_result = predict_one_image(
            stage_model,
            sample_image,
            trigger_type=trigger_type,
            target_label=target_label,
            trigger_size=int(stage_config.get("trigger_size", 4)),
            alpha=float(stage_config.get("alpha", 0.8)),
            position=str(stage_config.get("position", "bottom_right")),
        )
        save_json(
            {
                "sample_image": str(sample_image),
                "model_source": model_source,
                "model_source_reason": model_source_reason,
                "checkpoint_path": str(effective_checkpoint_path),
                "clean": {key: value for key, value in clean_result.items() if key != "processed_image"},
                "triggered": {key: value for key, value in triggered_result.items() if key != "processed_image"},
            },
            output_dir / "single_image_prediction.json",
        )

        eval_summary = evaluate_small_subset(
            model=stage_model,
            subset_test_dir=subset_test_dir,
            trigger_type=trigger_type,
            target_label=target_label,
            trigger_size=int(stage_config.get("trigger_size", 4)),
            alpha=float(stage_config.get("alpha", 0.8)),
            position=str(stage_config.get("position", "bottom_right")),
            batch_size=1,
        )
    else:
        eval_summary = _formal_eval_summary(
            experiment_name=experiment_name,
            trigger_type=trigger_type,
            target_label=target_label,
        )
    eval_summary.update(
        {
            "model_source": model_source,
            "model_source_reason": model_source_reason,
            "checkpoint_path": str(effective_checkpoint_path),
            "official_checkpoint_path": str(official_checkpoint_path),
            "demo_checkpoint_path": train_log["demo_checkpoint_path"],
            "epochs_completed": int(train_log["epochs_completed"]),
        }
    )
    save_json(eval_summary, output_dir / "demo_eval_summary.json")

    return {
        "output_dir": str(output_dir),
        "official_checkpoint_path": str(official_checkpoint_path),
        "checkpoint_path": str(effective_checkpoint_path),
        "model_source": model_source,
        "model_source_reason": model_source_reason,
        "clean_result": clean_result,
        "triggered_result": triggered_result,
        "eval_summary": eval_summary,
        "train_log": train_log,
        "demo_eval_summary_path": str(output_dir / "demo_eval_summary.json"),
        "demo_train_log_path": str(output_dir / "demo_train_log.json"),
        "demo_training_curve_csv_path": train_log["demo_training_curve_csv_path"],
        "training_curve_path": train_log["training_curve_path"],
    }
