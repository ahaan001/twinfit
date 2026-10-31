"""Printables, the metadata template and the protocol document."""

import io
from pathlib import Path

import cv2
import numpy as np
import yaml

from twinfit.contracts.conventions import aruco_dictionary, charuco_board
from twinfit.contracts.models import ClipMetadata
from twinfit.printables import write_metadata_template, write_printables
from twinfit.printables.pdfs import MM_PER_IN, PDF_NAMES, board_figure, part_markers_figure

REPO_ROOT = Path(__file__).resolve().parents[2]


def _render_png(fig, dpi: int) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)


def test_write_printables_creates_three_letter_pdfs(isolated_data_dir):
    written = write_printables()
    assert [p.name for p in written] == list(PDF_NAMES)
    for path in written:
        assert path.parent == isolated_data_dir / "printables"
        data = path.read_bytes()
        assert data.startswith(b"%PDF")
        # US Letter landscape in PDF points: 11 x 8.5 in at 72 pt/in.
        assert b"/MediaBox [ 0 0 792 612 ]" in data


def test_metadata_template_validates(isolated_data_dir):
    path = write_metadata_template()
    assert path == isolated_data_dir / "real" / "_template" / "metadata.yaml"
    text = path.read_text()
    assert "# " in text, "the template must keep its comments"
    meta = ClipMetadata.model_validate(yaml.safe_load(text))
    assert meta.source == "real"
    assert meta.gravity_W is None
    assert meta.intrinsics == "../_calibration/<camera_id>/camera_intrinsics.json"
    for field in ClipMetadata.model_fields:
        assert f"{field}:" in text, f"template is missing {field}"


def test_protocol_document_states_board_axes():
    text = (REPO_ROOT / "docs" / "TEST_DATA_PROTOCOL.md").read_text()
    assert "normal_toward_camera_W" in text
    assert "up_W = (0, -1, 0)" in text
    assert "~/twinfit-data/printables/" in text
    assert "~/twinfit-data/real/_template/metadata.yaml" in text


def test_board_drawn_at_physical_size_and_orientation():
    dpi = 100
    img = _render_png(board_figure(), dpi)
    assert img.shape == (int(8.5 * dpi), int(11 * dpi))
    board = charuco_board()
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    assert ids is not None and len(ids) == 24
    uv = corners.reshape(-1, 2).astype(float)
    xy = board.getChessboardCorners()[ids.ravel(), :2].astype(float)
    # Affine fit W (m) -> pixels; the diagonal is the printed scale in px per metre.
    design = np.hstack([xy, np.ones((len(xy), 1))])
    coef, _, _, _ = np.linalg.lstsq(design, uv, rcond=None)
    A = coef[:2].T
    px_per_m = dpi / (MM_PER_IN / 1000.0)
    square_px = 0.035 * px_per_m
    assert abs(A[0, 0] * 0.035 - square_px) < 0.01 * square_px
    assert abs(A[1, 1] * 0.035 - square_px) < 0.01 * square_px
    assert abs(A[0, 1]) < 1e-3 * px_per_m and abs(A[1, 0]) < 1e-3 * px_per_m
    # +x runs right and +y runs down the page, so the y = 0 edge is the top edge under the arrow.
    assert A[0, 0] > 0 and A[1, 1] > 0
    assert uv[:, 1].min() > 0.5 * dpi, "board must leave room for the UP arrow above it"


def test_part_markers_detectable_at_30_mm():
    dpi = 200
    img = _render_png(part_markers_figure(), dpi)
    corners, ids, _ = cv2.aruco.ArucoDetector(aruco_dictionary("DICT_4X4_50")).detectMarkers(img)
    assert ids is not None and sorted(ids.ravel().tolist()) == list(range(12))
    side_px = 30.0 / MM_PER_IN * dpi
    for quad in corners:
        side = np.linalg.norm(quad[0][1] - quad[0][0])
        assert abs(side - side_px) < 0.01 * side_px
