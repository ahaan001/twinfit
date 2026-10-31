"""Printable PDFs at exact physical size on US Letter landscape.

Every figure is 11 x 8.5 in with one axes spanning the whole page whose data units are inches,
so a length of L mm is drawn as L / 25.4 data units. Saving with `format="pdf"` and no bounding
box cropping keeps the page at exactly 11 x 8.5 in, which makes "Print at 100%" exact.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from twinfit import paths
from twinfit.contracts.conventions import (
    CHARUCO_DICTIONARY,
    CHARUCO_MARKER_LENGTH_M,
    CHARUCO_SQUARE_LENGTH_M,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    PART_MARKER_DICTIONARY,
    PART_MARKER_IDS,
    PART_MARKER_LENGTH_M,
    aruco_dictionary,
    charuco_board,
)

PAGE_W_IN = 11.0
PAGE_H_IN = 8.5
MM_PER_IN = 25.4
SCALE_BAR_MM = 100.0

BOARD_PX_PER_SQUARE = 140
BOARD_W_MM = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_M * 1000.0
BOARD_H_MM = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_M * 1000.0

PART_MARKER_MM = PART_MARKER_LENGTH_M * 1000.0
PART_MARKER_BORDER_MM = 6.0
PART_MARKER_PX = 120

SPECKLE_DOT_MM = (2.0, 4.0)
SPECKLE_COVERAGE = 0.35
SPECKLE_SEED = 0

PDF_DPI = 600

PDF_NAMES = ("charuco_board_letter.pdf", "part_markers_letter.pdf", "speckle_letter.pdf")


def mm(value_mm: float) -> float:
    return value_mm / MM_PER_IN


def _page() -> tuple[Figure, Axes]:
    fig = Figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, PAGE_W_IN)
    ax.set_ylim(0.0, PAGE_H_IN)
    ax.set_aspect("equal", adjustable="box")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


def _scale_bar(ax: Axes, x0: float, y: float) -> tuple[float, float]:
    """A 100 mm bar with end ticks. Returns its (x0, x1) in inches."""
    x1 = x0 + mm(SCALE_BAR_MM)
    tick = mm(2.5)
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.5, solid_capstyle="butt")
    for x in (x0, x1):
        ax.plot([x, x], [y - tick, y + tick], color="black", linewidth=1.5)
    for k in range(1, 10):
        xk = x0 + k * mm(10.0)
        ax.plot([xk, xk], [y, y + tick * 0.6], color="black", linewidth=0.8)
    ax.text((x0 + x1) / 2, y + mm(4.0), "100 mm scale bar", ha="center", va="bottom", fontsize=9)
    return x0, x1


def _check_page(fig: Figure, ax: Axes) -> None:
    fig.canvas.draw()
    assert tuple(fig.get_size_inches()) == (PAGE_W_IN, PAGE_H_IN)
    pos = ax.get_position()
    assert np.allclose([pos.x0, pos.y0, pos.x1, pos.y1], [0.0, 0.0, 1.0, 1.0])
    assert ax.get_xlim() == (0.0, PAGE_W_IN)
    assert ax.get_ylim() == (0.0, PAGE_H_IN)
    assert ax.get_aspect() == 1.0
    assert not ax.get_autoscale_on()


def _check_length(a: float, b: float, expected_mm: float) -> None:
    assert abs((b - a) - mm(expected_mm)) < 1e-9, f"{(b - a) * MM_PER_IN} mm != {expected_mm} mm"


def _imshow_binary(ax: Axes, img: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    # origin="upper" puts image row 0 at the top edge of the extent.
    ax.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )


def board_image() -> np.ndarray:
    board = charuco_board()
    size = (CHARUCO_SQUARES_X * BOARD_PX_PER_SQUARE, CHARUCO_SQUARES_Y * BOARD_PX_PER_SQUARE)
    return board.generateImage(size, marginSize=0, borderBits=1)


def board_extent_in() -> tuple[float, float, float, float]:
    """(left, right, bottom, top) of the centered board, in page inches."""
    w, h = mm(BOARD_W_MM), mm(BOARD_H_MM)
    left = (PAGE_W_IN - w) / 2
    bottom = (PAGE_H_IN - h) / 2
    return left, left + w, bottom, bottom + h


def board_figure() -> Figure:
    fig, ax = _page()
    left, right, bottom, top = board_extent_in()
    _imshow_binary(ax, board_image(), (left, right, bottom, top))
    ax.set_xlim(0.0, PAGE_W_IN)
    ax.set_ylim(0.0, PAGE_H_IN)

    # UP arrow above the board's top edge (image row 0, the y = 0 side of the board frame).
    arrow_x = PAGE_W_IN / 2
    arrow_base = top + mm(2.5)
    arrow_tip = PAGE_H_IN - mm(4.0)
    ax.annotate(
        "",
        xy=(arrow_x, arrow_tip),
        xytext=(arrow_x, arrow_base),
        arrowprops={
            "arrowstyle": "simple,head_width=1.6,head_length=1.2,tail_width=0.6",
            "color": "black",
        },
    )
    ax.text(
        arrow_x + mm(8.0),
        (arrow_base + arrow_tip) / 2,
        "UP",
        ha="left",
        va="center",
        fontsize=30,
        fontweight="bold",
    )
    ax.text(
        arrow_x - mm(8.0),
        (arrow_base + arrow_tip) / 2,
        "mount with this edge on top",
        ha="right",
        va="center",
        fontsize=9,
    )

    bar_y = mm(6.0)
    x0, x1 = _scale_bar(ax, left, bar_y)
    square_mm = CHARUCO_SQUARE_LENGTH_M * 1000.0
    marker_mm = CHARUCO_MARKER_LENGTH_M * 1000.0
    info = (
        f"ChArUco {CHARUCO_DICTIONARY}, {CHARUCO_SQUARES_X} x {CHARUCO_SQUARES_Y} squares, "
        f"square {square_mm:g} mm, marker {marker_mm:g} mm, "
        f"board {BOARD_W_MM:g} x {BOARD_H_MM:g} mm"
    )
    ax.text(right, bar_y + mm(4.0), info, ha="right", va="bottom", fontsize=9)
    ax.text(
        right,
        bar_y - mm(1.0),
        "Print at 100% (Actual size). Measure squares and record them in metadata.yaml.",
        ha="right",
        va="top",
        fontsize=9,
    )

    _check_page(fig, ax)
    _check_length(left, right, BOARD_W_MM)
    _check_length(bottom, top, BOARD_H_MM)
    _check_length(x0, x1, SCALE_BAR_MM)
    return fig


def part_marker_cells_in() -> list[tuple[int, float, float]]:
    """(marker id, left, bottom) of each marker's black square, in page inches."""
    columns, rows = 4, 3
    cell = mm(PART_MARKER_MM + 2 * PART_MARKER_BORDER_MM)
    pitch = cell + mm(6.0)
    grid_w = columns * pitch - mm(6.0)
    grid_h = rows * pitch - mm(6.0)
    grid_left = (PAGE_W_IN - grid_w) / 2
    grid_top = PAGE_H_IN - mm(12.0)
    cells = []
    for marker_id in PART_MARKER_IDS:
        col, row = marker_id % columns, marker_id // columns
        cell_left = grid_left + col * pitch
        cell_top = grid_top - row * pitch
        left = cell_left + mm(PART_MARKER_BORDER_MM)
        bottom = cell_top - cell + mm(PART_MARKER_BORDER_MM)
        cells.append((marker_id, left, bottom))
    assert grid_top - grid_h > mm(30.0), "marker grid must leave room for the scale bar"
    return cells


def part_markers_figure() -> Figure:
    fig, ax = _page()
    dictionary = aruco_dictionary(PART_MARKER_DICTIONARY)
    size = mm(PART_MARKER_MM)
    border = mm(PART_MARKER_BORDER_MM)
    for marker_id, left, bottom in part_marker_cells_in():
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, PART_MARKER_PX)
        _imshow_binary(ax, img, (left, left + size, bottom, bottom + size))
        _check_length(left, left + size, PART_MARKER_MM)
        ax.add_patch(
            Rectangle(
                (left - border, bottom - border),
                size + 2 * border,
                size + 2 * border,
                fill=False,
                edgecolor="black",
                linewidth=0.6,
                linestyle=(0, (4, 3)),
            )
        )
        # The id sits inside the white border so it stays with the marker after cutting.
        ax.text(
            left + size / 2,
            bottom - border + mm(1.0),
            f"id {marker_id}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_xlim(0.0, PAGE_W_IN)
    ax.set_ylim(0.0, PAGE_H_IN)

    grid_left = part_marker_cells_in()[0][1] - border
    ax.text(
        grid_left,
        mm(50.0),
        f"Part markers {PART_MARKER_DICTIONARY}, ids {PART_MARKER_IDS[0]} to "
        f"{PART_MARKER_IDS[-1]}, {PART_MARKER_MM:g} mm black square, "
        f"{PART_MARKER_BORDER_MM:g} mm white border",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        grid_left,
        mm(43.0),
        "Print at 100% (Actual size). Cut along the dashed lines and keep the white border.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        grid_left,
        mm(36.0),
        "Measure one black square and record it in metadata.yaml.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    x0, x1 = _scale_bar(ax, grid_left, mm(22.0))

    _check_page(fig, ax)
    _check_length(x0, x1, SCALE_BAR_MM)
    return fig


def speckle_region_in() -> tuple[float, float, float, float]:
    """(left, right, bottom, top) of the speckled area, in page inches."""
    margin = mm(12.0)
    return margin, PAGE_W_IN - margin, mm(32.0), PAGE_H_IN - margin


def speckle_figure() -> Figure:
    fig, ax = _page()
    left, right, bottom, top = speckle_region_in()
    d_lo, d_hi = SPECKLE_DOT_MM
    # Mean dot area for diameters uniform in [d_lo, d_hi]: E[d^2] = (lo^2 + lo*hi + hi^2) / 3.
    mean_dot_area_mm2 = np.pi / 4 * (d_lo**2 + d_lo * d_hi + d_hi**2) / 3
    region_area_mm2 = (right - left) * (top - bottom) * MM_PER_IN**2
    n_dots = int(round(SPECKLE_COVERAGE * region_area_mm2 / mean_dot_area_mm2))

    rng = np.random.default_rng(SPECKLE_SEED)
    diameters = rng.uniform(d_lo, d_hi, size=n_dots)
    xs = rng.uniform(left, right, size=n_dots)
    ys = rng.uniform(bottom, top, size=n_dots)
    dots = [Circle((x, y), mm(d) / 2) for x, y, d in zip(xs, ys, diameters, strict=True)]
    collection = PatchCollection(dots, facecolor="black", edgecolor="none")
    clip = Rectangle((left, bottom), right - left, top - bottom, transform=ax.transData)
    collection.set_clip_path(clip)
    ax.add_collection(collection, autolim=False)
    ax.set_xlim(0.0, PAGE_W_IN)
    ax.set_ylim(0.0, PAGE_H_IN)

    ax.text(
        right,
        mm(27.0),
        f"Speckle pattern: {n_dots} dots, {d_lo:g} to {d_hi:g} mm, "
        f"nominal {SPECKLE_COVERAGE:.0%} coverage, seed {SPECKLE_SEED}",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        right,
        mm(21.0),
        "Print at 100% (Actual size) on matte paper. Cut patches and tape at least two onto the "
        "camera-facing side of every moving part.",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    x0, x1 = _scale_bar(ax, left, mm(10.0))

    _check_page(fig, ax)
    _check_length(x0, x1, SCALE_BAR_MM)
    return fig


def write_printables(out_dir: Path | None = None) -> list[Path]:
    out = Path(out_dir) if out_dir is not None else paths.data_dir() / "printables"
    out.mkdir(parents=True, exist_ok=True)
    figures = (board_figure(), part_markers_figure(), speckle_figure())
    written = []
    for name, fig in zip(PDF_NAMES, figures, strict=True):
        path = out / name
        fig.savefig(path, format="pdf", dpi=PDF_DPI)
        written.append(path)
    return written
