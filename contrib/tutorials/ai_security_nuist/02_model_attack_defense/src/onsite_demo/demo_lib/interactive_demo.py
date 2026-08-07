from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from .inference import predict_one_image
from .paths import resolve_project_root
from .subset import list_image_records


SUPPORTED_IMAGE_SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg"}


def _class_name(label: int) -> str:
    return f"label {int(label):02d}"


def _source_dir(project_root: str | Path, source: str) -> Path:
    root = Path(project_root)
    source_key = str(source).strip().lower()
    mapping = {
        "data_test": root / "src" / "data" / "test",
        "data_train": root / "src" / "data" / "train",
        "demo_subset_test": root / "src" / "onsite_demo" / "outputs" / "demo_subset" / "test",
        "demo_subset_train": root / "src" / "onsite_demo" / "outputs" / "demo_subset" / "train",
    }
    if source_key not in mapping:
        raise ValueError("image_source 只能是 data_test、data_train、demo_subset_test 或 demo_subset_train。")
    directory = mapping[source_key]
    if not directory.exists():
        raise FileNotFoundError(f"图片来源目录不存在：{directory}")
    return directory


def _infer_label_from_path(image_path: Path, source_dir: Path) -> int:
    relative = image_path.relative_to(source_dir)
    label_text = relative.parts[0]
    if not label_text.isdigit():
        raise ValueError(f"无法从类别目录推断 label：{image_path}")
    return int(label_text)


def _collect_image_items_from_directory(
    source_dir: Path,
    *,
    source_name: str,
    max_images: int | None = None,
    skip_target_label: bool = False,
    target_label: int = 0,
) -> list[dict[str, Any]]:
    image_paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )

    items: list[dict[str, Any]] = []
    for image_path in image_paths:
        label = _infer_label_from_path(image_path, source_dir)
        if skip_target_label and label == int(target_label):
            continue
        items.append(
            {
                "image_path": str(image_path),
                "true_label": int(label),
                "class_name": _class_name(label),
                "relative_path": str(image_path.relative_to(source_dir)),
                "source": str(source_name),
                "index": len(items),
            }
        )
        if max_images is not None and len(items) >= int(max_images):
            break
    return items


def _parse_optional_seed(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(text)


def _draw_top5_bar(ax, top5: list[dict[str, Any]], title: str) -> None:
    labels = [str(item["label"]) for item in top5]
    values = [float(item["probability"]) for item in top5]
    positions = np.arange(len(labels))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Probability")
    ax.set_title(title)


def _json_safe_top5(top5: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": int(item["label"]), "probability": float(item["probability"])} for item in top5]


def _trigger_title(trigger_type: str) -> str:
    return "Triggered: Square" if trigger_type == "square" else "Triggered: Checkerboard"


def _select_model_for_trigger(trigger_type: str, square_model=None, checkerboard_model=None):
    trigger = str(trigger_type).strip().lower()
    if trigger == "square":
        if square_model is None:
            raise ValueError("square_model 尚未加载，请先运行 Square checkpoint 加载 cell。")
        return trigger, square_model
    if trigger == "checkerboard":
        if checkerboard_model is None:
            raise ValueError("checkerboard_model 尚未加载，请先运行 Checkerboard checkpoint 加载 cell。")
        return trigger, checkerboard_model
    raise ValueError("trigger_type 只能是 'square' 或 'checkerboard'。")


def _validate_models_for_widget(fixed_trigger_type: str | None, square_model=None, checkerboard_model=None) -> tuple[bool, str]:
    if fixed_trigger_type is None:
        missing: list[str] = []
        if square_model is None:
            missing.append("square_model 尚未加载，请先运行 Square checkpoint 加载 cell。")
        if checkerboard_model is None:
            missing.append("checkerboard_model 尚未加载，请先运行 Checkerboard checkpoint 加载 cell。")
        return (not missing, " ".join(missing))
    try:
        _select_model_for_trigger(fixed_trigger_type, square_model=square_model, checkerboard_model=checkerboard_model)
    except ValueError as exc:
        return False, str(exc)
    return True, ""


def collect_attack_images(
    project_root: str | Path,
    source: str = "data_test",
    max_images: int | None = None,
    skip_target_label: bool = False,
    target_label: int = 0,
) -> list[dict[str, Any]]:
    """Collect candidate images without checking whether the attack will succeed."""

    source_dir = _source_dir(project_root, source)
    items = _collect_image_items_from_directory(
        source_dir,
        source_name=str(source),
        max_images=max_images,
        skip_target_label=skip_target_label,
        target_label=target_label,
    )

    if not items:
        print(f"提示：{source} 中没有可用图片。")
    return items


def _resolve_detection_source_dir(
    project_root: str | Path,
    subset_test_dir: str | Path,
    image_source: str,
) -> tuple[Path, str]:
    source_key = str(image_source).strip().lower()
    if source_key == "data_test":
        return _source_dir(project_root, "data_test"), "data_test"
    if source_key in {"demo_subset", "demo_subset_test"}:
        subset_dir = Path(subset_test_dir)
        if not subset_dir.exists():
            raise FileNotFoundError(f"subset_test_dir does not exist: {subset_dir}")
        return subset_dir, "demo_subset"
    raise ValueError("image_source must be 'data_test' or 'demo_subset'.")


def _select_detection_bundle(trigger_type: str, detection_model_bundles: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    trigger = str(trigger_type).strip().lower()
    if trigger not in {"square", "checkerboard"}:
        raise ValueError("trigger_type must be 'square' or 'checkerboard'.")
    if trigger not in detection_model_bundles:
        raise ValueError(f"Missing detection model bundle for trigger_type={trigger!r}.")
    bundle = dict(detection_model_bundles[trigger])
    if bundle.get("model") is None:
        raise ValueError(f"Detection model for trigger_type={trigger!r} is not loaded.")
    return trigger, bundle


def run_free_detection_test(
    *,
    project_root: str | Path,
    subset_test_dir: str | Path,
    detection_model_bundles: dict[str, dict[str, Any]],
    detection_config: dict[str, Any],
    image_source: str = "data_test",
    random_pick: bool = True,
    image_index: int = 0,
    trigger_type: str = "square",
    skip_target_label: bool = True,
    random_seed: int | None = None,
    target_label: int = 0,
    k: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    from .detection import run_strip_light_demo

    trigger, bundle = _select_detection_bundle(trigger_type, detection_model_bundles)
    detection_batch_size = int(detection_config.get("strip_light_batch_size", 1))
    source_dir, normalized_source = _resolve_detection_source_dir(project_root, subset_test_dir, image_source)
    image_items = _collect_image_items_from_directory(
        source_dir,
        source_name=normalized_source,
        skip_target_label=skip_target_label,
        target_label=target_label,
    )
    if not image_items:
        raise ValueError(f"No candidate images found for source={normalized_source}.")

    selected_seed = _parse_optional_seed(random_seed)
    if bool(random_pick):
        item, selected_index = select_random_image(image_items, seed=selected_seed)
    else:
        item = select_image_by_index(image_items, image_index)
        selected_index = int(image_index)

    result = run_strip_light_demo(
        model=bundle["model"],
        subset_test_dir=Path(subset_test_dir),
        trigger_type=trigger,
        target_label=target_label,
        k=int(k),
        reference_count=int(detection_config["strip_light_reference_count"]),
        calibration_count=int(detection_config["strip_light_calibration_count"]),
        target_fpr=float(detection_config["strip_light_target_fpr"]),
        trigger_size=int(bundle.get("trigger_size", 4)),
        alpha=float(bundle.get("alpha", 0.8)),
        position=str(bundle.get("position", "bottom_right")),
        batch_size=detection_batch_size,
        seed=int(seed),
        candidate_image_path=item["image_path"],
        model_source=bundle.get("model_source"),
    )
    result.update(
        {
            "image_source": normalized_source,
            "image_index": int(selected_index),
            "relative_path": str(item.get("relative_path", "")),
            "random_pick": bool(random_pick),
            "skip_target_label": bool(skip_target_label),
            "k": int(k),
        }
    )
    return result


def show_strip_light_detection_result(result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import display

    from .detection_visualization import (
        plot_strip_light_target_probability,
        show_strip_light_blend_grid,
        show_strip_light_candidate_pair,
        show_strip_light_score_card,
        show_strip_light_summary_table,
    )

    figures = [
        show_strip_light_candidate_pair(result),
        show_strip_light_blend_grid(result),
        plot_strip_light_target_probability(result),
        show_strip_light_score_card(result),
        show_strip_light_summary_table(result),
    ]
    for fig in figures:
        display(fig)
        plt.close(fig)
    return None


def show_attack_image_gallery(image_items: list[dict[str, Any]], max_images: int = 24, cols: int = 6):
    """Show a candidate image gallery only when the notebook explicitly asks for it."""

    import matplotlib.pyplot as plt
    from PIL import Image

    items = image_items[: int(max_images)]
    if not items:
        print("没有候选图片可展示。")
        return None

    cols = max(1, min(int(cols), len(items)))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.35))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")

    for display_index, item in enumerate(items):
        with Image.open(item["image_path"]) as image:
            axes_array[display_index].imshow(image.convert("RGB"))
        axes_array[display_index].set_title(
            f"idx {int(item['index'])}\nlabel {int(item['true_label']):02d}",
            fontsize=9,
        )
        axes_array[display_index].axis("off")

    fig.suptitle("Candidate Images", fontsize=13)
    plt.tight_layout()
    return fig


def select_image_by_index(image_items: list[dict[str, Any]], image_index: int) -> dict[str, Any]:
    if not image_items:
        raise ValueError("候选图片列表为空，无法按 index 选择图片。")
    index = int(image_index)
    if index < 0 or index >= len(image_items):
        raise IndexError(f"IMAGE_INDEX={index} 越界，合法范围是 0 到 {len(image_items) - 1}。")
    return image_items[index]


def select_random_image(image_items: list[dict[str, Any]], seed: int | None = None) -> tuple[dict[str, Any], int]:
    if not image_items:
        raise ValueError("候选图片列表为空，无法随机选择图片。")
    rng = random.Random(seed) if seed is not None else random.Random()
    selected_index = rng.randrange(len(image_items))
    return image_items[selected_index], selected_index


def _predict_clean_and_triggered(
    image_path: str | Path,
    trigger_type: str,
    model,
    target_label: int,
    trigger_size: int,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trigger = str(trigger_type).strip().lower()
    if trigger not in {"square", "checkerboard"}:
        raise ValueError("trigger_type 只能是 'square' 或 'checkerboard'。")

    clean = predict_one_image(
        model,
        image_path,
        trigger_type=None,
        target_label=target_label,
        trigger_size=trigger_size,
        alpha=alpha,
    )
    triggered = predict_one_image(
        model,
        image_path,
        trigger_type=trigger,
        target_label=target_label,
        trigger_size=trigger_size,
        alpha=alpha,
    )
    return clean, triggered


def _build_single_trigger_result(
    item: dict[str, Any],
    image_index: int,
    image_source: str,
    trigger_type: str,
    model,
    target_label: int,
    trigger_size: int,
    alpha: float,
    skip_target_label_warning: bool = True,
) -> dict[str, Any]:
    clean, triggered = _predict_clean_and_triggered(
        image_path=item["image_path"],
        trigger_type=trigger_type,
        model=model,
        target_label=target_label,
        trigger_size=trigger_size,
        alpha=alpha,
    )
    true_label = int(item["true_label"])
    true_label_is_target = true_label == int(target_label)
    warning_message = ""
    if skip_target_label_warning and true_label_is_target:
        warning_message = "当前图片真实标签已经是 target label，不适合判断 ASR，可重新随机选择一张非 target 图片。"

    return {
        "image_source": str(image_source),
        "image_index": int(image_index),
        "image_path": str(item["image_path"]),
        "relative_path": str(item.get("relative_path", "")),
        "true_label": true_label,
        "trigger_type": str(trigger_type).strip().lower(),
        "target_label": int(target_label),
        "clean_prediction": int(clean["pred_label"]),
        "clean_confidence": float(clean["probability"]),
        "clean_top5": _json_safe_top5(clean["top5"]),
        "triggered_prediction": int(triggered["pred_label"]),
        "triggered_confidence": float(triggered["probability"]),
        "triggered_target_confidence": float(triggered["target_probability"]),
        "triggered_top5": _json_safe_top5(triggered["top5"]),
        "attack_success": int(triggered["pred_label"]) == int(target_label),
        "true_label_is_target": true_label_is_target,
        "warning_message": warning_message,
        "clean_image": clean["processed_image"],
        "triggered_image": triggered["processed_image"],
    }


def run_single_trigger_attack_test(
    image_source: str,
    image_index: int,
    trigger_type: str,
    model,
    target_label: int,
    project_root: str | Path | None = None,
    skip_target_label_warning: bool = True,
    trigger_size: int = 4,
    alpha: float = 0.8,
) -> dict[str, Any]:
    """Run clean-vs-triggered inference for one manually selected image."""

    root = Path(project_root) if project_root is not None else resolve_project_root()["PROJECT_ROOT"]
    image_items = collect_attack_images(root, source=image_source, skip_target_label=False, target_label=target_label)
    item = select_image_by_index(image_items, image_index)
    print(
        f"已选择图片 source={image_source} index={int(image_index)} "
        f"true_label={item['true_label']} path={item['relative_path']}"
    )
    return _build_single_trigger_result(
        item=item,
        image_index=int(image_index),
        image_source=image_source,
        trigger_type=trigger_type,
        model=model,
        target_label=target_label,
        trigger_size=trigger_size,
        alpha=alpha,
        skip_target_label_warning=skip_target_label_warning,
    )


def run_random_attack_test(
    image_source: str,
    trigger_type: str,
    model,
    target_label: int,
    project_root: str | Path | None = None,
    seed: int | None = None,
    skip_target_label: bool = True,
    trigger_size: int = 4,
    alpha: float = 0.8,
) -> dict[str, Any]:
    """Randomly choose one image and test exactly one trigger type."""

    root = Path(project_root) if project_root is not None else resolve_project_root()["PROJECT_ROOT"]
    image_items = collect_attack_images(
        root,
        source=image_source,
        skip_target_label=skip_target_label,
        target_label=target_label,
    )
    item, selected_index = select_random_image(image_items, seed=seed)
    print(
        f"随机选择图片 source={image_source} index={selected_index} "
        f"true_label={item['true_label']} path={item['relative_path']}"
    )
    result = _build_single_trigger_result(
        item=item,
        image_index=selected_index,
        image_source=image_source,
        trigger_type=trigger_type,
        model=model,
        target_label=target_label,
        trigger_size=trigger_size,
        alpha=alpha,
        skip_target_label_warning=True,
    )
    result.update(
        {
            "random_pick": True,
            "random_seed": seed,
            "candidate_count": len(image_items),
            "skip_target_label": bool(skip_target_label),
        }
    )
    return result


def show_single_trigger_attack_result(result: dict[str, Any], save_path: str | Path | None = None) -> None:
    """Show one compact figure for clean vs triggered predictions."""

    import matplotlib.pyplot as plt

    trigger_type = str(result["trigger_type"]).strip().lower()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    clean_ax = axes[0, 0]
    trigger_ax = axes[0, 1]
    clean_bar_ax = axes[1, 0]
    trigger_bar_ax = axes[1, 1]

    clean_ax.imshow(np.clip(result["clean_image"], 0.0, 1.0))
    clean_ax.set_title("Clean")
    clean_ax.set_xlabel(f"pred={int(result['clean_prediction'])} prob={float(result['clean_confidence']):.3f}")
    clean_ax.set_xticks([])
    clean_ax.set_yticks([])

    trigger_ax.imshow(np.clip(result["triggered_image"], 0.0, 1.0))
    trigger_ax.set_title(_trigger_title(trigger_type))
    trigger_ax.set_xlabel(
        f"pred={int(result['triggered_prediction'])} prob={float(result['triggered_confidence']):.3f}\n"
        f"target_conf={float(result['triggered_target_confidence']):.3f}"
    )
    trigger_ax.set_xticks([])
    trigger_ax.set_yticks([])
    if bool(result["attack_success"]):
        for spine in trigger_ax.spines.values():
            spine.set_edgecolor("green")
            spine.set_linewidth(3)

    _draw_top5_bar(clean_bar_ax, result["clean_top5"], "Clean Top-5")
    _draw_top5_bar(trigger_bar_ax, result["triggered_top5"], f"{trigger_type.title()} Top-5")
    fig.suptitle(
        f"true={int(result['true_label'])} target={int(result['target_label'])} "
        f"success={bool(result['attack_success'])}",
        fontsize=12,
    )
    plt.tight_layout()

    print(f"true label：{int(result['true_label'])}")
    print(f"clean prediction：{int(result['clean_prediction'])}  confidence={float(result['clean_confidence']):.4f}")
    print(
        f"triggered prediction：{int(result['triggered_prediction'])}  "
        f"confidence={float(result['triggered_confidence']):.4f}"
    )
    print(f"target label：{int(result['target_label'])}")
    print(f"target confidence：{float(result['triggered_target_confidence']):.4f}")
    print(f"attack success：{bool(result['attack_success'])}")
    if result.get("warning_message"):
        print(f"提示：{result['warning_message']}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return None


def run_free_attack_test(
    project_root: str | Path | None = None,
    square_model=None,
    checkerboard_model=None,
    image_source: str = "data_test",
    random_pick: bool = True,
    image_index: int = 0,
    trigger_type: str = "square",
    skip_target_label: bool = True,
    random_seed: int | None = None,
    target_label: int = 0,
    trigger_size: int = 4,
    alpha: float = 0.8,
    **legacy_kwargs,
) -> dict[str, Any]:
    """Variable-control helper used by notebook cells and widget callbacks."""

    image_source = legacy_kwargs.pop("IMAGE_SOURCE", image_source)
    random_pick = legacy_kwargs.pop("RANDOM_PICK", random_pick)
    image_index = legacy_kwargs.pop("IMAGE_INDEX", image_index)
    trigger_type = legacy_kwargs.pop("TRIGGER_TYPE", trigger_type)
    skip_target_label = legacy_kwargs.pop("SKIP_TARGET_LABEL", skip_target_label)
    random_seed = legacy_kwargs.pop("RANDOM_SEED", random_seed)
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"run_free_attack_test 收到未知参数：{unexpected}")

    _select_model_for_trigger(
        trigger_type,
        square_model=square_model,
        checkerboard_model=checkerboard_model,
    )
    trigger = str(trigger_type).strip().lower()
    if trigger == "square":
        model = square_model
    elif trigger == "checkerboard":
        model = checkerboard_model
    else:
        raise ValueError("trigger_type 只能是 'square' 或 'checkerboard'。")
    if model is None:
        raise ValueError(f"{trigger} 模型尚未加载，无法运行攻击测试。")

    seed = _parse_optional_seed(random_seed)
    if bool(random_pick):
        return run_random_attack_test(
            image_source=image_source,
            trigger_type=trigger,
            model=model,
            target_label=target_label,
            project_root=project_root,
            seed=seed,
            skip_target_label=skip_target_label,
            trigger_size=trigger_size,
            alpha=alpha,
        )
    return run_single_trigger_attack_test(
        image_source=image_source,
        image_index=int(image_index),
        trigger_type=trigger,
        model=model,
        target_label=target_label,
        project_root=project_root,
        skip_target_label_warning=True,
        trigger_size=trigger_size,
        alpha=alpha,
    )


def run_widget_demo_if_available(
    project_root: str | Path,
    square_model=None,
    checkerboard_model=None,
    target_label: int = 0,
    default_image_source: str = "data_test",
    default_trigger_type: str = "square",
    fixed_trigger_type: str | None = None,
    enable_widgets: bool = True,
):
    """Display a real ipywidgets control panel when ipywidgets is installed.

    Returns a dict with widget handles on success. If ipywidgets is unavailable,
    returns ``False, reason`` and does not raise.
    """

    if not enable_widgets:
        return False, "控件未启用。"
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)

    models_ok, model_reason = _validate_models_for_widget(
        fixed_trigger_type,
        square_model=square_model,
        checkerboard_model=checkerboard_model,
    )
    if not models_ok:
        return False, model_reason

    source_dropdown = widgets.Dropdown(
        options=[("data_test", "data_test"), ("demo_subset_test", "demo_subset_test")],
        value=default_image_source,
        description="Source",
        layout=widgets.Layout(width="260px"),
    )
    index_input = widgets.BoundedIntText(
        value=0,
        min=0,
        max=999999,
        step=1,
        description="Index",
        layout=widgets.Layout(width="220px"),
    )
    random_checkbox = widgets.Checkbox(value=True, description="Random pick")
    skip_target_checkbox = widgets.Checkbox(value=True, description="Skip target label")
    run_button = widgets.Button(description="运行攻击测试", button_style="primary", icon="play")
    output = widgets.Output()
    index_row = widgets.HBox([index_input])

    def _sync_index_visibility() -> None:
        index_row.layout.display = "none" if bool(random_checkbox.value) else ""

    def run_once(_=None):  # noqa: ANN001
        with output:
            clear_output(wait=True)
            try:
                trigger_value = str(fixed_trigger_type or default_trigger_type)
                if fixed_trigger_type is None:
                    trigger_value = str(default_trigger_type if trigger_dropdown is None else trigger_dropdown.value)
                result = run_free_attack_test(
                    project_root=project_root,
                    square_model=square_model,
                    checkerboard_model=checkerboard_model,
                    image_source=str(source_dropdown.value),
                    random_pick=bool(random_checkbox.value),
                    image_index=int(index_input.value),
                    trigger_type=trigger_value,
                    skip_target_label=bool(skip_target_checkbox.value),
                    random_seed=None,
                    target_label=target_label,
                )
                show_single_trigger_attack_result(result)
            except Exception as exc:  # noqa: BLE001
                print(f"攻击测试运行失败：{exc}")

    trigger_dropdown = None
    trigger_row = None
    if fixed_trigger_type is None:
        trigger_dropdown = widgets.Dropdown(
            options=[("square", "square"), ("checkerboard", "checkerboard")],
            value=default_trigger_type,
            description="Trigger",
            layout=widgets.Layout(width="260px"),
        )
        trigger_row = widgets.HBox([trigger_dropdown])

    random_checkbox.observe(lambda change: _sync_index_visibility(), names="value")
    _sync_index_visibility()
    run_button.on_click(run_once)
    header_items = [source_dropdown]
    if trigger_dropdown is not None:
        header_items.append(trigger_dropdown)
    controls = widgets.VBox(
        [
            widgets.HBox(header_items),
            widgets.HBox([random_checkbox, skip_target_checkbox, run_button]),
            index_row,
            output,
        ]
    )
    display(controls)
    return {
        "source_dropdown": source_dropdown,
        "trigger_dropdown": trigger_dropdown,
        "index_input": index_input,
        "index_row": index_row,
        "random_checkbox": random_checkbox,
        "skip_target_checkbox": skip_target_checkbox,
        "run_button": run_button,
        "output": output,
    }


def run_detection_widget_demo_if_available(
    *,
    project_root: str | Path,
    subset_test_dir: str | Path,
    detection_model_bundles: dict[str, dict[str, Any]],
    detection_config: dict[str, Any],
    target_label: int = 0,
    default_image_source: str = "data_test",
    default_trigger_type: str = "square",
    default_k: int = 8,
    seed: int = 42,
    enable_widgets: bool = True,
    result_output_dir: str | Path | None = None,
):
    if not enable_widgets:
        return False, "widgets disabled"
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)

    for required_trigger in ("square", "checkerboard"):
        _select_detection_bundle(required_trigger, detection_model_bundles)

    source_dropdown = widgets.Dropdown(
        options=[("data_test", "data_test"), ("demo_subset", "demo_subset")],
        value=default_image_source,
        description="Source",
        layout=widgets.Layout(width="240px"),
    )
    trigger_dropdown = widgets.Dropdown(
        options=[("square", "square"), ("checkerboard", "checkerboard")],
        value=default_trigger_type,
        description="Trigger",
        layout=widgets.Layout(width="260px"),
    )
    random_checkbox = widgets.Checkbox(value=True, description="Random pick")
    skip_target_checkbox = widgets.Checkbox(value=True, description="Skip target label")
    index_input = widgets.BoundedIntText(
        value=0,
        min=0,
        max=999999,
        step=1,
        description="Image index",
        layout=widgets.Layout(width="220px"),
    )
    k_dropdown = widgets.Dropdown(
        options=[("4", 4), ("8", 8), ("10", 10), ("16", 16)],
        value=int(default_k),
        description="K",
        layout=widgets.Layout(width="180px"),
    )
    run_button = widgets.Button(description="运行检测", button_style="primary", icon="search")
    output = widgets.Output()
    index_row = widgets.HBox([index_input])

    def _sync_index_visibility() -> None:
        index_row.layout.display = "none" if bool(random_checkbox.value) else ""

    def run_once(_=None):  # noqa: ANN001
        with output:
            clear_output(wait=True)
            try:
                result = run_free_detection_test(
                    project_root=project_root,
                    subset_test_dir=subset_test_dir,
                    detection_model_bundles=detection_model_bundles,
                    detection_config=detection_config,
                    image_source=str(source_dropdown.value),
                    random_pick=bool(random_checkbox.value),
                    image_index=int(index_input.value),
                    trigger_type=str(trigger_dropdown.value),
                    skip_target_label=bool(skip_target_checkbox.value),
                    random_seed=None,
                    target_label=target_label,
                    k=int(k_dropdown.value),
                    seed=int(seed),
                )
                print(f"selected source = {result.get('image_source')}")
                print(f"selected trigger = {result.get('trigger_type')}")
                print(f"selected index = {result.get('image_index')}")
                print(f"selected image = {result.get('relative_path')}")
                print(f"model_source = {result.get('model_source')}")
                if result_output_dir is not None:
                    from .detection import save_strip_light_result

                    output_dir = Path(result_output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    result_path = save_strip_light_result(
                        result,
                        output_dir / f"{result.get('trigger_type', 'unknown')}_strip_light_result.notebook.json",
                    )
                    print(f"result path = {result_path}")
                show_strip_light_detection_result(result)
            except Exception as exc:  # noqa: BLE001
                print(f"Detection demo failed: {exc}")

    random_checkbox.observe(lambda change: _sync_index_visibility(), names="value")
    _sync_index_visibility()
    run_button.on_click(run_once)
    controls = widgets.VBox(
        [
            widgets.HBox([source_dropdown, trigger_dropdown, k_dropdown]),
            widgets.HBox([random_checkbox, skip_target_checkbox, run_button]),
            index_row,
            output,
        ]
    )
    display(controls)
    return {
        "source_dropdown": source_dropdown,
        "trigger_dropdown": trigger_dropdown,
        "random_checkbox": random_checkbox,
        "skip_target_checkbox": skip_target_checkbox,
        "index_input": index_input,
        "index_row": index_row,
        "k_dropdown": k_dropdown,
        "run_button": run_button,
        "output": output,
    }


# Compatibility helpers for earlier rehearsal notebooks.
def list_demo_images(subset_test_dir: str | Path, target_label: int = 0, max_per_class: int = 5) -> list[dict[str, Any]]:
    subset_test_dir = Path(subset_test_dir)
    records = list_image_records(subset_test_dir)
    per_class_counts: dict[int, int] = {}
    image_items: list[dict[str, Any]] = []
    for image_path, label in records:
        label = int(label)
        if label == int(target_label) or per_class_counts.get(label, 0) >= int(max_per_class):
            continue
        per_class_counts[label] = per_class_counts.get(label, 0) + 1
        image_items.append(
            {
                "image_path": str(image_path),
                "true_label": label,
                "class_name": _class_name(label),
                "relative_path": str(image_path.relative_to(subset_test_dir)),
                "source": "demo_subset_test",
                "index": len(image_items),
            }
        )
    return image_items


def build_image_gallery(image_items: list[dict[str, Any]], cols: int = 6, max_images: int = 24):
    return show_attack_image_gallery(image_items, max_images=max_images, cols=cols)
