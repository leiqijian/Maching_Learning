from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def find_data_file(data_path: str | None) -> Path:
    if data_path:
        path = Path(data_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"找不到数据文件: {path}")

    csv_files = sorted(Path(".").glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("当前目录没有找到 CSV 文件。")
    return csv_files[0]


def load_dataset(data_file: Path, max_rows: int | None) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_file, encoding="utf-8-sig", nrows=max_rows)

    if "label" not in df.columns:
        raise ValueError("CSV 中未找到 label 列，请确认数据格式为 label + pixel0..pixel783。")

    y = df["label"].to_numpy()
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return x, y


def preprocess_demo_image(image_path: Path) -> np.ndarray:
    arr = plt.imread(image_path)
    arr = np.asarray(arr, dtype=np.float32)

    # plt.imread 读取 PNG 时常返回 0~1 浮点，这里统一拉回 0~255。
    if arr.max() <= 1.0:
        arr = arr * 255.0

    # RGB/RGBA 转灰度
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)

    # 自适应阈值二值化：对细笔画更稳，不容易被灰度抗锯齿干扰。
    threshold = max(30.0, float(arr.mean() + 0.5 * arr.std()))
    binary = (arr > threshold).astype(np.uint8) * 255

    ys, xs = np.where(binary > 0)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((1, 28 * 28), dtype=np.float32)

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = binary[y0 : y1 + 1, x0 : x1 + 1]

    # 按 MNIST 常见处理将数字主体缩放到 20x20 内，再居中放置到 28x28 画布。
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    canvas = np.zeros((28, 28), dtype=np.float32)
    off_y = (28 - new_h) // 2
    off_x = (28 - new_w) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

    return canvas.reshape(1, -1)


def read_demo_grayscale(image_path: Path) -> np.ndarray:
    arr = plt.imread(image_path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def show_images(image_path: Path, processed: np.ndarray, pred: int, confidence: float) -> None:
    original = read_demo_grayscale(image_path)
    processed_2d = processed.reshape(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("原图")
    axes[0].axis("off")

    axes[1].imshow(processed_2d, cmap="gray")
    axes[1].set_title(f"预处理后\n预测={pred}, conf={confidence:.3f}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def resolve_input_path(input_path: str) -> Path:
    p = Path(input_path)
    if p.exists():
        return p

    p2 = Path(str(input_path).lstrip("\\/"))
    if p2.exists():
        return p2

    p3 = Path(__file__).resolve().parent / str(input_path).lstrip("\\/")
    if p3.exists():
        return p3

    raise FileNotFoundError(f"找不到输入路径: {input_path}")


def predict_one_image(model: KNeighborsClassifier, image_file: Path) -> tuple[int, float, str, np.ndarray, int, float, int, float]:
    demo_raw = read_demo_grayscale(image_file).reshape(1, -1)
    demo_centered = preprocess_demo_image(image_file)
    pred_raw = int(model.predict(demo_raw)[0])
    pred_centered = int(model.predict(demo_centered)[0])
    conf_raw = float(np.max(model.predict_proba(demo_raw)[0]))
    conf_centered = float(np.max(model.predict_proba(demo_centered)[0]))

    if conf_centered > conf_raw + 1e-6:
        return pred_centered, conf_centered, "centered", demo_centered, pred_raw, conf_raw, pred_centered, conf_centered
    return pred_raw, conf_raw, "raw", demo_raw, pred_raw, conf_raw, pred_centered, conf_centered


def predict_with_vote(
    model: KNeighborsClassifier, image_file: Path
) -> tuple[int, float, str, np.ndarray, int, float, int, float]:
    base_arr = read_demo_grayscale(image_file)

    variants: list[np.ndarray] = []
    # 平移增强：缓解数字在画布中位置偏差。
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            shifted = np.roll(np.roll(base_arr, dy, axis=0), dx, axis=1)
            variants.append(shifted)

    # 轻微膨胀：缓解细笔画与训练集分布差异。
    pil_img = Image.fromarray(base_arr.astype(np.uint8))
    variants.append(np.asarray(pil_img.filter(ImageFilter.MaxFilter(3)), dtype=np.float32))
    variants.append(np.asarray(pil_img.filter(ImageFilter.MaxFilter(5)), dtype=np.float32))

    preds: list[int] = []
    probas: list[np.ndarray] = []
    for arr in variants:
        x = arr.reshape(1, -1)
        preds.append(int(model.predict(x)[0]))
        probas.append(model.predict_proba(x)[0])

    counter = Counter(preds)
    voted_pred, voted_count = counter.most_common(1)[0]
    vote_ratio = voted_count / len(preds)

    # 从投票结果里挑一个同类最高置信度样本作为展示输入。
    best_idx = None
    best_conf = -1.0
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        conf = float(np.max(proba))
        if pred == voted_pred and conf > best_conf:
            best_conf = conf
            best_idx = i

    assert best_idx is not None
    chosen_x = variants[best_idx].reshape(1, -1)

    # 保留 raw/centered 指标，方便对比和排查。
    final_pred, final_conf, chosen, _demo_x, pred_raw, conf_raw, pred_centered, conf_centered = predict_one_image(
        model, image_file
    )
    _ = (final_pred, final_conf, chosen, _demo_x)

    return (
        int(voted_pred),
        float(vote_ratio),
        "vote",
        chosen_x,
        pred_raw,
        conf_raw,
        pred_centered,
        conf_centered,
    )


def main(image_path: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="KNN 手写数字识别示例")
    parser.add_argument("image_path", nargs="?", default=None, help="待识别图片路径")
    parser.add_argument("--data", default=None, help="CSV 数据路径（默认自动寻找当前目录第一个 csv）")
    parser.add_argument("--neighbors", type=int, default=1, help="KNN 的 k 值，默认 1")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例，默认 0.2")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="读取数据行数上限（默认 0 表示读取全部）",
    )
    parser.add_argument("--show", action="store_true", help="显示原图和预处理后的图像")
    args = parser.parse_args()

    data_file = find_data_file(args.data)

    final_input_path = image_path if image_path is not None else args.image_path
    if not final_input_path:
        raise ValueError("请在 main(image_path=...) 中传入图片或文件夹路径，或通过命令行传入 image_path 参数。")

    input_path = resolve_input_path(final_input_path)

    max_rows = None if args.max_rows == 0 else args.max_rows
    x, y = load_dataset(data_file, max_rows=max_rows)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model = KNeighborsClassifier(n_neighbors=args.neighbors, weights="distance")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"数据文件: {data_file}")
    print(f"训练样本数: {len(x_train)}, 测试样本数: {len(x_test)}")
    print(f"验证集准确率: {acc:.4f}")
    if input_path.is_dir():
        image_files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        )
        if not image_files:
            raise FileNotFoundError(f"文件夹中没有可识别图片: {input_path}")
        print(f"识别文件夹: {input_path}")
        for img in image_files:
            demo_pred, confidence, chosen, _demo_x, pred_raw, conf_raw, pred_centered, conf_centered = predict_with_vote(
                model, img
            )
            print(f"{img.name} -> 预测={demo_pred}, 置信度={confidence:.4f}, 路径={chosen}, raw={pred_raw}/{conf_raw:.4f}, centered={pred_centered}/{conf_centered:.4f}")
    else:
        image_file = input_path
        demo_pred, confidence, chosen, demo_x, pred_raw, conf_raw, pred_centered, conf_centered = predict_with_vote(
            model, image_file
        )
        print(f"识别图片: {image_file}")
        print(f"原图预测: {pred_raw} (conf={conf_raw:.4f}), 居中预测: {pred_centered} (conf={conf_centered:.4f})")
        print(f"最终采用: {chosen}")
        print(f"预测结果: {demo_pred}")
        print(f"置信度: {confidence:.4f}")
        if args.show:
            show_images(image_file, demo_x, int(demo_pred), confidence)


if __name__ == "__main__":
    # 支持传单个图片路径，也支持传文件夹路径（批量识别）。
    main(image_path="num")
