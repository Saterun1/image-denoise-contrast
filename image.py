#!/usr/bin/env python3
"""Базовый пайплайн обработки изображения.

Что делает скрипт:
1. Загружает цветное изображение и переводит его в пространство LAB.
2. Добавляет синтетический гауссов шум к каналу яркости L
   (sigma = 5...40).
3. Обрабатывает зашумлённый канал L несколькими методами:
   Gaussian, Median, HE (глобальное выравнивание гистограммы),
   CLAHE, а также практические комбинации Gaussian+HE, Gaussian+CLAHE,
   Median+HE и Median+CLAHE.
4. Считает PSNR и SSIM относительно чистого канала L.
5. Сохраняет CSV с метриками, график PSNR/SSIM от уровня шума
   и наглядную монтажную сетку результатов.
6. Дополнительно сохраняет улучшенную версию исходного изображения
   (лёгкое шумоподавление + CLAHE).

Запуск:
    python image.py
    python image.py --input "F:/i.webp" --output-dir output
    python image.py --sigmas 5 10 15 20 25 30 35 40 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")  # без GUI: скрипт работает и в PyCharm, и в консоли

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEFAULT_SIGMAS = (5, 10, 15, 20, 25, 30, 35, 40)
DEFAULT_SEED = 42
DATA_RANGE = 255.0

Method = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Загрузка / преобразование изображения
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    """Читает изображение через Pillow (поддерживает webp, png, jpg и т.д.)."""
    if not path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {path}")
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return rgb


def rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает каналы L, a, b изображения в пространстве LAB (uint8)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    return l_channel, a_channel, b_channel


def lab_to_rgb(
    l_channel: np.ndarray, a_channel: np.ndarray, b_channel: np.ndarray
) -> np.ndarray:
    """Собирает RGB-изображение из канала яркости L и исходных a/b."""
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def add_gaussian_noise(
    channel: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Добавляет аддитивный гауссов шум с заданным стандартным отклонением."""
    noise = rng.normal(loc=0.0, scale=sigma, size=channel.shape)
    noisy = channel.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Методы обработки
# ---------------------------------------------------------------------------
def make_clahe(clip_limit: float, tile_grid: tuple[int, int]) -> cv2.CLAHE:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)


def build_methods(
    clip_limit: float,
    tile_grid: tuple[int, int],
    gaussian_ksize: int,
    median_ksize: int,
) -> dict[str, Method]:
    """Создаёт словарь методов: имя -> функция обработки одноканального L."""
    clahe = make_clahe(clip_limit, tile_grid)

    def gaussian(channel: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(channel, (gaussian_ksize, gaussian_ksize), 0)

    def median(channel: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(channel, median_ksize)

    def equalize(channel: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(channel)

    def clahe_apply(channel: np.ndarray) -> np.ndarray:
        return clahe.apply(channel)

    def gaussian_he(channel: np.ndarray) -> np.ndarray:
        return equalize(gaussian(channel))

    def median_he(channel: np.ndarray) -> np.ndarray:
        return equalize(median(channel))

    def gaussian_clahe(channel: np.ndarray) -> np.ndarray:
        return clahe.apply(gaussian(channel))

    def median_clahe(channel: np.ndarray) -> np.ndarray:
        return clahe.apply(median(channel))

    return {
        "Gaussian": gaussian,
        "Median": median,
        "HE": equalize,
        "CLAHE": clahe_apply,
        "Gaussian+HE": gaussian_he,
        "Gaussian+CLAHE": gaussian_clahe,
        "Median+HE": median_he,
        "Median+CLAHE": median_clahe,
    }


# ---------------------------------------------------------------------------
# Метрики и эксперимент
# ---------------------------------------------------------------------------
def compute_metrics(reference: np.ndarray, processed: np.ndarray) -> tuple[float, float]:
    """PSNR (dB) и SSIM относительно опорного изображения."""
    psnr = float(
        peak_signal_noise_ratio(reference, processed, data_range=DATA_RANGE)
    )
    ssim = float(
        structural_similarity(reference, processed, data_range=DATA_RANGE)
    )
    return psnr, ssim


def evaluate(
    clean_l: np.ndarray,
    sigmas: Iterable[int],
    methods: dict[str, Method],
    seed: int,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    """Прогоняет все методы на всех уровнях шума.

    Возвращает строки для CSV и зашумлённые L-каналы (для монтажа).
    Для каждого sigma создаётся один и тот же зашумлённый канал,
    который затем обрабатывается всеми методами.
    """
    rows: list[dict[str, object]] = []
    noisy_by_sigma: dict[int, np.ndarray] = {}

    for sigma in sigmas:
        # Отдельный генератор на каждый уровень шума: результат
        # воспроизводим при фиксированном seed.
        rng = np.random.default_rng(seed + int(sigma))
        noisy_l = add_gaussian_noise(clean_l, sigma, rng)
        noisy_by_sigma[int(sigma)] = noisy_l

        for name, method in methods.items():
            started = time.perf_counter()
            processed = method(noisy_l)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            psnr, ssim = compute_metrics(clean_l, processed)
            rows.append(
                {
                    "method": name,
                    "sigma": int(sigma),
                    "psnr_db": round(psnr, 4),
                    "ssim": round(ssim, 6),
                    "time_ms": round(elapsed_ms, 3),
                }
            )

    return rows, noisy_by_sigma


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------
def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["method", "sigma", "psnr_db", "ssim", "time_ms"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, object]], path: Path) -> None:
    """Строит два графика: PSNR(sigma) и SSIM(sigma)."""
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    sigmas = sorted({int(row["sigma"]) for row in rows})

    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for index, method in enumerate(methods):
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["sigma"]),
        )
        x = [int(row["sigma"]) for row in method_rows]
        psnr = [float(row["psnr_db"]) for row in method_rows]
        ssim = [float(row["ssim"]) for row in method_rows]

        style = {
            "color": colors[index % len(colors)],
            "marker": markers[index % len(markers)],
            "markersize": 6,
            "linewidth": 1.8,
            "label": method,
        }
        axes[0].plot(x, psnr, **style)
        axes[1].plot(x, ssim, **style)

    axes[0].set_title("PSNR: зависимость от уровня шума")
    axes[0].set_xlabel("Стандартное отклонение шума sigma")
    axes[0].set_ylabel("PSNR, дБ")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("SSIM: зависимость от уровня шума")
    axes[1].set_xlabel("Стандартное отклонение шума sigma")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.3)

    for axis in axes:
        axis.set_xticks(sigmas)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )
    fig.suptitle(
        "Сравнение шумоподавления и контраста на синтетическом гауссовом шуме",
        fontsize=14,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _panel_title(
    rows: list[dict[str, object]], method: str, sigma: int
) -> str:
    for row in rows:
        if row["method"] == method and int(row["sigma"]) == sigma:
            return f"{method}\nPSNR {float(row['psnr_db']):.2f} dB | SSIM {float(row['ssim']):.3f}"
    return method


def save_montage(
    clean_rgb: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    noisy_l: np.ndarray,
    rows: list[dict[str, object]],
    methods: dict[str, Method],
    sigma: int,
    path: Path,
) -> None:
    """Сохраняет сетку: Clean / Noisy / результаты всех методов."""
    panels: list[tuple[str, np.ndarray]] = [
        ("Clean", clean_rgb),
        (f"Noisy (sigma={sigma})", lab_to_rgb(noisy_l, a_channel, b_channel)),
    ]

    for name, method in methods.items():
        processed_l = method(noisy_l)
        panels.append(
            (
                _panel_title(rows, name, sigma),
                lab_to_rgb(processed_l, a_channel, b_channel),
            )
        )

    ncols = 5 if len(panels) >= 10 else 4
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), constrained_layout=True
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, (title, image) in zip(axes_flat, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    for axis in axes_flat[len(panels) :]:
        axis.axis("off")

    fig.suptitle(
        f"Визуальное сравнение методов при sigma={sigma}", fontsize=14
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_enhanced(
    clean_rgb: np.ndarray,
    clean_l: np.ndarray,
    a_channel: np.ndarray,
    b_channel: np.ndarray,
    clip_limit: float,
    tile_grid: tuple[int, int],
    output_dir: Path,
) -> None:
    """Лёгкое шумоподавление + CLAHE для исходного (чистого) изображения."""
    clahe = make_clahe(clip_limit, tile_grid)
    denoised_l = cv2.medianBlur(clean_l, 3)
    enhanced_l = clahe.apply(denoised_l)
    enhanced_rgb = lab_to_rgb(enhanced_l, a_channel, b_channel)

    Image.fromarray(enhanced_rgb).save(output_dir / "enhanced.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].imshow(clean_rgb)
    axes[0].set_title("Исходное изображение")
    axes[0].axis("off")
    axes[1].imshow(enhanced_rgb)
    axes[1].set_title("Median 3x3 + CLAHE")
    axes[1].axis("off")
    fig.savefig(output_dir / "original_vs_enhanced.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Шумоподавление и контраст: Gaussian/Median/HE/CLAHE + PSNR/SSIM."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input.webp"),
        help="Путь к исходному изображению (по умолчанию data/input.webp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Каталог для результатов (по умолчанию output).",
    )
    parser.add_argument(
        "--sigmas",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIGMAS),
        help="Уровни гауссова шума sigma (по умолчанию 5 10 ... 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed генератора шума (по умолчанию 42).",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="clipLimit для CLAHE (по умолчанию 2.0).",
    )
    parser.add_argument(
        "--tile-grid",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("TX", "TY"),
        help="Сетка тайлов CLAHE (по умолчанию 8 8).",
    )
    parser.add_argument(
        "--gaussian-ksize",
        type=int,
        default=5,
        help="Размер ядра Gaussian (нечётный, по умолчанию 5).",
    )
    parser.add_argument(
        "--median-ksize",
        type=int,
        default=5,
        help="Размер ядра Median (нечётный, по умолчанию 5).",
    )
    parser.add_argument(
        "--montage-sigma",
        type=int,
        default=25,
        help="Уровень шума для визуальной сетки (по умолчанию 25).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.sigmas or any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("Уровни шума sigma должны быть положительными.")
    for name, value in (
        ("gaussian-ksize", args.gaussian_ksize),
        ("median-ksize", args.median_ksize),
    ):
        if value < 3 or value % 2 == 0:
            raise ValueError(f"--{name} должен быть нечётным числом >= 3.")
    if args.clip_limit <= 0:
        raise ValueError("--clip-limit должен быть > 0.")
    if any(size <= 0 for size in args.tile_grid):
        raise ValueError("--tile-grid должен содержать положительные числа.")


def print_summary(rows: list[dict[str, object]]) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    sigmas = sorted({int(row["sigma"]) for row in rows})

    print("\nЛучший метод по PSNR/SSIM на каждом уровне шума:")
    print(f"{'sigma':>6} | {'best PSNR':<18} | {'best SSIM':<18}")
    print("-" * 50)
    for sigma in sigmas:
        sigma_rows = [row for row in rows if int(row["sigma"]) == sigma]
        best_psnr = max(sigma_rows, key=lambda row: float(row["psnr_db"]))
        best_ssim = max(sigma_rows, key=lambda row: float(row["ssim"]))
        print(
            f"{sigma:>6} | "
            f"{str(best_psnr['method']):<10} {float(best_psnr['psnr_db']):>6.2f} | "
            f"{str(best_ssim['method']):<10} {float(best_ssim['ssim']):>6.3f}"
        )

    print("\nСредние метрики по всем уровням шума:")
    print(f"{'method':<18} | {'mean PSNR':>10} | {'mean SSIM':>10}")
    print("-" * 45)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        mean_psnr = np.mean([float(row["psnr_db"]) for row in method_rows])
        mean_ssim = np.mean([float(row["ssim"]) for row in method_rows])
        print(f"{method:<18} | {mean_psnr:>10.2f} | {mean_ssim:>10.3f}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    args = parse_args()
    validate_args(args)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка: {args.input}")
    rgb = load_rgb(args.input)
    clean_l, a_channel, b_channel = rgb_to_lab(rgb)
    print(f"Размер изображения: {rgb.shape[1]}x{rgb.shape[0]}")

    methods = build_methods(
        clip_limit=args.clip_limit,
        tile_grid=tuple(args.tile_grid),
        gaussian_ksize=args.gaussian_ksize,
        median_ksize=args.median_ksize,
    )

    print(f"Уровни шума sigma: {', '.join(map(str, args.sigmas))}")
    sigmas = sorted(set(args.sigmas))
    rows, noisy_by_sigma = evaluate(
        clean_l=clean_l,
        sigmas=sigmas,
        methods=methods,
        seed=args.seed,
    )

    save_csv(rows, output_dir / "metrics.csv")
    plot_metrics(rows, output_dir / "psnr_ssim_vs_noise.png")

    montage_sigma = min(sigmas, key=lambda sigma: abs(sigma - args.montage_sigma))
    save_montage(
        clean_rgb=rgb,
        a_channel=a_channel,
        b_channel=b_channel,
        noisy_l=noisy_by_sigma[montage_sigma],
        rows=rows,
        methods=methods,
        sigma=montage_sigma,
        path=output_dir / f"comparison_sigma{montage_sigma}.png",
    )

    save_enhanced(
        clean_rgb=rgb,
        clean_l=clean_l,
        a_channel=a_channel,
        b_channel=b_channel,
        clip_limit=args.clip_limit,
        tile_grid=tuple(args.tile_grid),
        output_dir=output_dir,
    )

    print_summary(rows)
    print("\nСохранено:")
    print(f"  {output_dir / 'metrics.csv'}")
    print(f"  {output_dir / 'psnr_ssim_vs_noise.png'}")
    print(f"  {output_dir / f'comparison_sigma{montage_sigma}.png'}")
    print(f"  {output_dir / 'enhanced.png'}")
    print(f"  {output_dir / 'original_vs_enhanced.png'}")


if __name__ == "__main__":
    main()
