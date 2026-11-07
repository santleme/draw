"""Grade de sessao, sugestao optica e escala dos formatos em retrato."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app, config, ink, paper  # noqa: E402
from draw.telemetry import SessionTelemetry  # noqa: E402
from tools import export_vectors as ev  # noqa: E402


@pytest.fixture(autouse=True)
def restore_a4():
    config.apply_format("a4")
    yield
    config.apply_format("a4")


@pytest.mark.parametrize("name,width", [
    ("a4", 850), ("a5", 846), ("a3", 849), ("carta", 929), ("caderno", 873),
])
def test_paper_grid(name, width):
    assert config.paper_grid(name) == (width, 1200)
    w_mm, h_mm = config.FORMATS[name]
    error = abs(width / 1200 - w_mm / h_mm)
    isotropy = abs(config.mm_per_px(name) / (w_mm / width) - 1)
    if name == "a4":
        # A unica excecao e a grade legada, nao uma tolerancia geral maior.
        assert error == pytest.approx(abs(850 / 1200 - 210 / 297))
        assert isotropy == pytest.approx(0.001785714285714)
    else:
        assert width == round(1200 * w_mm / h_mm)
        assert error <= 0.001
        assert isotropy <= 0.001


def test_apply_format_a4_eh_incumbente():
    original = {0: (90, 90), 1: (760, 90), 2: (760, 1110), 3: (90, 1110)}
    config.apply_format("carta")
    assert config.apply_format("a4") == (850, 1200)
    assert config.MARKER_CENTERS == original
    assert config.PAPER_FORMAT == "a4"
    frame = np.arange(200 * 300 * 3, dtype=np.uint8).reshape(200, 300, 3)
    quad = np.float32([[10, 10], [290, 10], [290, 190], [10, 190]])
    matrix = cv2.getPerspectiveTransform(
        quad, np.float32([[0, 0], [850, 0], [850, 1200], [0, 1200]])
    )
    np.testing.assert_array_equal(
        paper.warp_sheet(frame, quad),
        cv2.warpPerspective(frame, matrix, (850, 1200)),
    )


def test_apply_format_recomputa_marcadores():
    config.apply_format("carta")
    assert config.MARKER_CENTERS[1] == (929 - 90, 90)
    assert config.MARKER_CENTERS[2] == (929 - 90, 1200 - 90)
    assert config.mm_per_px() == 279 / 1200
    with pytest.raises(ValueError):
        config.apply_format("paisagem")
    assert config.PAPER_FORMAT == "carta"


def test_consumidores_seguem_a_grade():
    assert app._blank_sheet().shape == (1200, 850, 3)
    config.apply_format("carta")
    frame = np.full((300, 300, 3), 255, np.uint8)
    quad = np.float32([[10, 10], [290, 10], [290, 290], [10, 290]])
    assert paper.warp_sheet(frame, quad).shape == (1200, 929, 3)
    assert ink.InkAccumulator().state.shape == (1200, 929)
    assert ink.render_canvas(np.zeros((1200, 929), np.uint8)).shape == (1200, 929, 3)
    assert app._blank_sheet().shape == (1200, 929, 3)


def projected_quad(name="carta", pitch=25, yaw=15, roll=0):
    w, h = config.FORMATS[name]
    points = np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0],
                       [w/2, h/2, 0], [-w/2, h/2, 0]])
    x, y, z = np.deg2rad([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    camera = points @ (rz @ ry @ rx).T + [0, 0, 1000]
    return camera[:, :2] / camera[:, 2:] * (1.2 * 1280) + [640, 360]


@pytest.mark.parametrize("name,label", [
    ("carta", "carta"), ("a4", "serie_a"), ("a3", "serie_a"),
    ("a5", "serie_a"), ("caderno", "caderno"),
])
@pytest.mark.parametrize("pitch,yaw", [(25, 15), (0, 0)])
def test_estimador_sugere_classe_certa(name, label, pitch, yaw):
    result = paper.estimate_format(projected_quad(name, pitch, yaw), (720, 1280))
    expected = config.FORMATS[name][0] / config.FORMATS[name][1]
    assert result["ratio"] == pytest.approx(expected, rel=0.02)
    assert result["suggestion"] == label
    assert result["reason"] is None


def test_estimador_se_abstem_com_um_vp_no_infinito():
    result = paper.estimate_format(projected_quad(pitch=25, yaw=0), (720, 1280))
    assert result["suggestion"] is None
    assert result["reason"] == "single_vp_at_infinity"


@pytest.mark.parametrize("points", [
    [[0, 0], [1, 1], [2, 2], [3, 3]], [[1, 1]] * 4,
    [[0, 0], [1, 0], [np.nan, 1], [0, 1]],
])
def test_estimador_se_abstem_com_quad_degenerado(points):
    result = paper.estimate_format(points, (720, 1280))
    assert result["suggestion"] is None
    assert result["reason"] == "degenerate_quad"


def test_estimador_se_abstem_com_focal_invalida():
    points = projected_quad()
    points[:, 0] = 1280 - points[:, 0]
    # A mesma camera observada com principal deslocado viola as hipoteses.
    result = paper.estimate_format(points + [5000, 2000], (720, 1280))
    assert result["suggestion"] is None
    assert result["reason"] == "invalid_focal_length"


def test_estimador_ratio_ambiguo_e_invariante_a_translacao_e_escala():
    ratio = np.sqrt((210 / 297) * (200 / 275))
    quad = np.array([[0, 0], [ratio, 0], [ratio, 1], [0, 1]]) * 300
    for scale, shift in [(1, [0, 0]), (0.5, [300, 200])]:
        result = paper.estimate_format(quad * scale + shift, (720, 1280))
        assert result["reason"] == "ambiguous_ratio"


def test_sugestao_usa_mediana_de_15_e_nao_aplica(tmp_path, capsys):
    config.apply_format("caderno")
    suggester = paper.FormatSuggester()
    quad = projected_quad()
    for i in range(14):
        sample = quad + [300, 100] if i < 3 else quad
        assert suggester.observe(sample, (720, 1280)) is None
    result = suggester.observe(quad, (720, 1280))
    assert result["suggestion"] == "carta"
    assert suggester.observe(quad, (720, 1280)) is None
    path = tmp_path / "events.jsonl"
    telemetry = SessionTelemetry(path, interval=3600).start()
    app._report_format_suggestion(telemetry, result)
    telemetry.close()
    record, = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["event"] == "format_suggestion"
    assert record["suggestion"] == "carta"
    assert "carta" in capsys.readouterr().out
    assert config.PAPER_FORMAT == "caderno"


def test_sugestao_se_abstem_para_centros_de_marcadores():
    suggester = paper.FormatSuggester()
    result = None
    for _ in range(15):
        result = suggester.observe(projected_quad(), (720, 1280), sheet_boundary=False)
    assert result["suggestion"] is None
    assert result["reason"] == "marker_centers_not_sheet_boundary"


@pytest.mark.parametrize("flags,env,expected", [
    ([], None, "a4"), ([], "carta", "carta"),
    (["--format", "a5"], "carta", "a5"),
    (["--format", "a4"], "invalido", "a4"),
])
def test_cli_resolve_formato_antes_de_abrir_sessao(monkeypatch, flags, env, expected):
    if env is None:
        monkeypatch.delenv("DRAW_FORMAT", raising=False)
    else:
        monkeypatch.setenv("DRAW_FORMAT", env)
    monkeypatch.setattr(sys, "argv", ["main.py", *flags])
    called = []

    def run(**kwargs):
        called.append(kwargs)
        assert config.PAPER_FORMAT == expected
        assert (config.PAPER_W, config.PAPER_H) == config.paper_grid(expected)

    monkeypatch.setattr(app, "run", run)
    app.main()
    assert len(called) == 1


def test_cli_rejeita_env_invalido_antes_de_modelos(monkeypatch):
    monkeypatch.setenv("DRAW_FORMAT", "paisagem")
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(app, "run", lambda **kw: pytest.fail("abriu a sessao"))
    with pytest.raises(SystemExit, match="2"):
        app.main()


def test_abertura_da_sessao_registra_formato(tmp_path, monkeypatch):
    config.apply_format("carta")

    class NoFrames:
        def __init__(self, capture):
            pass

        def start(self):
            return self

        def wait_next(self, seq):
            return None

        def stop(self):
            pass

    monkeypatch.delenv("DRAW_NEURAL_MODEL", raising=False)
    monkeypatch.setattr(app, "_open_capture", lambda source: None)
    monkeypatch.setattr(app, "_LatestFrameCapture", NoFrames)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    app.run(save_dir=tmp_path, legacy_ink=True)
    records = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    opened, = [r for r in records if r.get("event") == "session_open"]
    assert opened["paper_format"] == "carta"
    assert opened["page_grid"] == [929, 1200]
    assert opened["mm_per_px"] == 279 / 1200
    report = json.loads((tmp_path / "session-reports/latest.json").read_text())
    assert report["paper_format"] == "carta"


@pytest.mark.parametrize("name", config.FORMATS)
def test_export_usa_escala_config_em_ambos_os_eixos(tmp_path, name):
    w, h = config.paper_grid(name)
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.line(img, (100, 100), (300, 300), (0, 0, 0), 3)
    result = ev.export_vectors(img, mask_source="otsu", paper_format=name)
    unit = config.mm_per_px(name)
    assert result["scale_mm_per_px"] == (unit, unit)
    paths = [{"points": np.float64([[100, 100], [300, 300]]), "closed": False}]
    dimensions = config.FORMATS[name]
    svg, dxf = tmp_path / "out.svg", tmp_path / "out.dxf"
    ev.write_svg(paths, svg, w, h, *dimensions, 1.0)
    ev.write_dxf(paths, dxf, w, h, *dimensions, 1.0)
    points_svg = ev.parse_svg_paths(svg)[0][0]
    points_dxf = ev.parse_dxf_polylines(dxf)[0][0]
    np.testing.assert_allclose(points_svg, np.array([[100, 100], [300, 300]]) * unit, atol=.006)
    np.testing.assert_allclose(points_dxf[:, 0], points_svg[:, 0], atol=.006)
    np.testing.assert_allclose(points_dxf[:, 1], dimensions[1] - points_svg[:, 1], atol=.006)
