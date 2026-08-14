from __future__ import annotations

import csv
import html
import os
from pathlib import Path


def plot_metrics_svg(
    metrics_paths: list[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    *,
    split: str = "validation",
    x_axis: str = "iteration",
) -> None:
    """Render comparable loss curves as a dependency-free SVG file."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if x_axis not in {"iteration", "elapsed_seconds"}:
        raise ValueError("x_axis must be 'iteration' or 'elapsed_seconds'")

    series: list[tuple[str, list[tuple[float, float]]]] = []
    for metrics_path in metrics_paths:
        path = Path(metrics_path)
        with open(path, encoding="utf-8") as metrics_file:
            rows = csv.DictReader(metrics_file)
            points = [(float(row[x_axis]), float(row["loss"])) for row in rows if row["split"] == split]
        if points:
            series.append((path.parent.name or path.stem, points))
    if not series:
        raise ValueError(f"no {split} metrics found")

    width, height = 900, 520
    left, right, top, bottom = 80, 30, 35, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for _, points in series for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1
    y_padding = 0.05 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    def coordinates(point: tuple[float, float]) -> tuple[float, float]:
        x = left + (point[0] - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - point[1]) / (y_max - y_min) * plot_height
        return x, y

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_width
        x_value = x_min + fraction * (x_max - x_min)
        y = top + fraction * plot_height
        y_value = y_max - fraction * (y_max - y_min)
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-size="12">{x_value:.3g}</text>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{y_value:.3f}</text>',
            ]
        )

    for index, (name, points) in enumerate(series):
        color = colors[index % len(colors)]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(coordinates, points))
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = top + 18 * index
        lines.append(f'<line x1="{left + 15}" y1="{legend_y}" x2="{left + 35}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{left + 42}" y="{legend_y + 4}" font-size="12">{html.escape(name)}</text>'
        )

    x_label = "wall-clock seconds" if x_axis == "elapsed_seconds" else "iteration"
    lines.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-size="14">{x_label}</text>',
            f'<text x="18" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {top + plot_height / 2})">{split} loss</text>',
            "</svg>",
        ]
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
