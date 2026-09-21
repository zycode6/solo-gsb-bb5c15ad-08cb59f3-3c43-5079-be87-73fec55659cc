"""凝胶迁移距离分子量校准：锚点枚举优选、回归残差与禁外推分段插值的单元 + API 端到端测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import SizeCalibrationRequest
from app.size_calibration import calibrate_sizes, _linear_regression
from app.validation import (
    DigestValidationError,
    validate_size_calibration_request,
)

client = TestClient(app)


def _sizes(distances, intercept=3.0, slope=-0.01):
    """按 log10(bp) = intercept + slope·d 生成严格降序的标准尺寸。"""
    return [round(10 ** (intercept + slope * d), 6) for d in distances]


def _req(ladder_sizes=None, ladder_distances=None, sample_distances=None,
         max_missing_or_spurious_bands=2, residual_threshold=0.01):
    return SizeCalibrationRequest(
        ladder_sizes=_sizes([10, 30, 55, 80, 110, 140]) if ladder_sizes is None else ladder_sizes,
        ladder_distances=[10, 30, 55, 80, 110, 140] if ladder_distances is None else ladder_distances,
        sample_distances=[] if sample_distances is None else sample_distances,
        max_missing_or_spurious_bands=max_missing_or_spurious_bands,
        residual_threshold=residual_threshold,
    )


# ---------------------------------------------------------------- 线性回归（纯函数）

def test_linear_regression_perfect_line():
    xs = [10.0, 30.0, 55.0, 80.0]
    ys = [3.0 - 0.01 * x for x in xs]
    slope, intercept, rmse, r_squared = _linear_regression(xs, ys)
    assert slope == pytest.approx(-0.01)
    assert intercept == pytest.approx(3.0)
    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert r_squared == pytest.approx(1.0, abs=1e-12)


def test_linear_regression_two_points_rmse_zero():
    # 两个点恒可被直线完全拟合，RMSE 为 0
    _, _, rmse, r_squared = _linear_regression([0.0, 100.0], [3.0, 2.0])
    assert rmse == 0.0
    assert r_squared == pytest.approx(1.0)


# ---------------------------------------------------------------- 锚点对齐（纯函数）

def test_perfect_ladder_all_anchors_and_linearity():
    res = calibrate_sizes(_req(sample_distances=[20, 140, 5, 150, 55]))
    assert res.anchor_count == 6
    assert res.skip_total == 0
    assert res.missing_standard_count == 0
    assert res.spurious_band_count == 0
    assert res.regression.slope == pytest.approx(-0.01, abs=1e-6)
    assert res.regression.rmse < 1e-5
    assert res.regression.within_threshold is True

    # 锚点按迁移距离升序（log10(bp) 降序）
    dists = [a.ladder_distance for a in res.anchors]
    sizes = [a.ladder_size for a in res.anchors]
    assert dists == sorted(dists)
    assert sizes == sorted(sizes, reverse=True)
    # 提交序号（两侧都是顺序提交）
    assert [(a.ladder_size_index, a.ladder_distance_index) for a in res.anchors] == [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    ]


def test_missing_standard_band_skips_standard_side():
    # 实测缺少 d=55（对应第 3 条标准带）；删除该标准带后仍严格线性，残差最小
    res = calibrate_sizes(_req(
        ladder_distances=[10, 30, 80, 110, 140],
        sample_distances=[55], max_missing_or_spurious_bands=1))
    assert res.anchor_count == 5
    assert res.missing_standard_count == 1
    assert res.spurious_band_count == 0
    assert res.skip_total == 1
    missing = res.missing_standard_bands
    assert [m.ladder_size_index for m in missing] == [3]
    assert missing[0].ladder_size == pytest.approx(_sizes([55])[0])

    # d=55 落在第 2、3 个入选锚点（d=30、d=80）之间插值
    band = res.sample_bands[0]
    assert band.status == "calibrated"
    assert band.segment_anchor_indexes == [2, 3]
    assert band.estimated_size == pytest.approx(round(10 ** (3 - 0.55), 2))


def test_spurious_band_skips_measured_side():
    # d=42 偏离线性关系，跳过它（杂带 1 条）即可得到完全线性的 6 锚点对齐
    res = calibrate_sizes(_req(
        ladder_distances=[10, 30, 42, 55, 80, 110, 140],
        max_missing_or_spurious_bands=2, residual_threshold=0.05))
    assert res.anchor_count == 6
    assert res.spurious_band_count == 1
    assert res.missing_standard_count == 0
    sp = res.spurious_bands[0]
    assert sp.distance == 42
    assert sp.ladder_distance_index == 3  # 请求提交序号


def test_lex_tiebreak_picks_smallest_first_diverging_pair():
    # 6 条实测全部共线、标准只有 5 条：必跳 1 条实测；跳过任意一条 RMSE 都约为 0，
    # 第三级索引序列字典序决定结果——首个分歧配对最小者胜，即跳过中间的 d=42
    # （对齐 (10,30,55,...) 的首分歧配对 (2,2) 小于对齐 (10,30,42,...) 的 (3,2)）
    res = calibrate_sizes(_req(
        ladder_sizes=_sizes([10, 30, 55, 80, 110]),
        ladder_distances=[10, 30, 42, 55, 80, 110],
        max_missing_or_spurious_bands=2, residual_threshold=0.05))
    assert res.anchor_count == 5
    assert [s.distance for s in res.spurious_bands] == [42]


def test_skip_total_beats_residual():
    # 一级目标优先：skip=1（残差略高但达标）胜过残差更低但 skip=2 的对齐
    res = calibrate_sizes(_req(
        ladder_sizes=_sizes([10, 30, 55, 80, 110]),
        ladder_distances=[10, 30, 46, 55, 80, 110],
        max_missing_or_spurious_bands=2, residual_threshold=0.2))
    assert res.anchor_count == 5
    assert res.skip_total == 1
    assert [s.distance for s in res.spurious_bands] == [46]


def test_result_independent_of_ladder_distance_submission_order():
    distances = [10, 30, 42, 55, 80, 110, 140]
    r1 = calibrate_sizes(_req(ladder_distances=distances,
                              max_missing_or_spurious_bands=2, residual_threshold=0.05))
    r2 = calibrate_sizes(_req(ladder_distances=list(reversed(distances)),
                              max_missing_or_spurious_bands=2, residual_threshold=0.05))
    key = lambda r: (
        r.skip_total,
        [(a.ladder_distance, a.ladder_size) for a in r.anchors],
        sorted(s.distance for s in r.spurious_bands),
    )
    assert key(r1) == key(r2)
    # 但提交序号随各自请求位置变化
    assert r1.spurious_bands[0].ladder_distance_index == 3
    assert r2.spurious_bands[0].ladder_distance_index == 5


# ---------------------------------------------------------------- 残差与锚点不足拒绝

def test_residual_exceeded_raises_422_level_error():
    with pytest.raises(DigestValidationError) as exc:
        calibrate_sizes(_req(
            ladder_sizes=[1000, 500, 100, 50, 20, 10],
            ladder_distances=[10, 30, 55, 80, 110, 140],
            max_missing_or_spurious_bands=4, residual_threshold=0.001))
    assert exc.value.detail["errors"][0]["type"] == "residual_threshold_exceeded"


def test_skip_budget_unreachable_raises():
    # n=2, m=4：即使最大对齐也至少跳过 2 项，超过预算 1
    with pytest.raises(DigestValidationError) as exc:
        calibrate_sizes(_req(
            ladder_sizes=[1000, 500],
            ladder_distances=[10, 20, 30, 40],
            max_missing_or_spurious_bands=1, residual_threshold=1.0))
    assert exc.value.detail["errors"][0]["type"] == "skip_limit_unreachable"


def test_residual_threshold_equal_is_accepted():
    # 阈值边界含等号：完美线性 ladder 的 RMSE ≈ 0，以 0 为阈值时仍通过
    res = calibrate_sizes(_req(residual_threshold=0.0))
    assert res.regression.within_threshold is True
    assert res.regression.residual_threshold == 0.0


# ---------------------------------------------------------------- 分段插值与禁外推

def test_piecewise_interpolation_and_no_extrapolation():
    res = calibrate_sizes(_req(
        ladder_sizes=[1000, 100], ladder_distances=[0, 100],
        sample_distances=[0, 50, 100],
        max_missing_or_spurious_bands=0, residual_threshold=1.0))
    by_dist = {b.distance: b for b in res.sample_bands}
    assert by_dist[0].status == "calibrated"
    assert by_dist[0].estimated_size == 1000.0
    assert by_dist[0].segment_anchor_indexes == [1]
    # log 空间中点：log10(bp)=2.5 → 几何均值 316.23 bp
    mid = by_dist[50]
    assert mid.status == "calibrated"
    assert mid.segment_anchor_indexes == [1, 2]
    assert mid.estimated_size == pytest.approx(316.23)
    assert by_dist[100].estimated_size == 100.0
    assert by_dist[100].segment_anchor_indexes == [2]

    assert res.calibration_range_min_distance == 0
    assert res.calibration_range_max_distance == 100
    assert res.calibration_range_min_size == 100.0
    assert res.calibration_range_max_size == 1000.0


def test_out_of_range_bands_not_extrapolated():
    res = calibrate_sizes(_req(
        sample_distances=[5, 150],
        max_missing_or_spurious_bands=0, residual_threshold=1.0))
    above, below = res.sample_bands
    assert above.status == "out_of_range_above"
    assert above.estimated_size is None
    assert above.segment_anchor_indexes is None
    assert below.status == "out_of_range_below"
    assert below.estimated_size is None
    assert below.segment_anchor_indexes is None


def test_sample_bands_follow_submission_order():
    res = calibrate_sizes(_req(sample_distances=[150, 20, 10]))
    assert [b.sample_band_index for b in res.sample_bands] == [1, 2, 3]
    assert [b.distance for b in res.sample_bands] == [150, 20, 10]
    assert [b.status for b in res.sample_bands] == [
        "out_of_range_below", "calibrated", "calibrated",
    ]


# ---------------------------------------------------------------- 语义校验（422）

def _validate(payload_override):
    base = dict(
        ladder_sizes=_sizes([10, 30, 55, 80, 110, 140]),
        ladder_distances=[10, 30, 55, 80, 110, 140],
        sample_distances=[],
        max_missing_or_spurious_bands=2,
        residual_threshold=0.05,
    )
    base.update(payload_override)
    return validate_size_calibration_request(SizeCalibrationRequest(**base))


def test_422_ladder_sizes_not_descending():
    with pytest.raises(DigestValidationError) as exc:
        _validate({"ladder_sizes": [1000, 500, 600, 100]})
    assert exc.value.detail["errors"][0]["type"] == "ladder_sizes_not_descending"


def test_422_duplicate_ladder_size_in_descending_check():
    with pytest.raises(DigestValidationError) as exc:
        _validate({"ladder_sizes": [1000, 500, 500, 100]})
    assert exc.value.detail["errors"][0]["type"] == "ladder_sizes_not_descending"


def test_422_duplicate_ladder_distance():
    with pytest.raises(DigestValidationError) as exc:
        _validate({"ladder_distances": [10, 30, 30, 80, 110, 140]})
    assert exc.value.detail["errors"][0]["type"] == "duplicate_distance"


def test_422_duplicate_sample_distance():
    with pytest.raises(DigestValidationError) as exc:
        _validate({"sample_distances": [50, 50]})
    assert exc.value.detail["errors"][0]["type"] == "duplicate_distance"


def test_422_too_many_ladder_bands(monkeypatch):
    from app.validation import settings
    monkeypatch.setattr(settings, "MAX_CALIBRATION_LADDER_BANDS", 4)
    with pytest.raises(DigestValidationError) as exc:
        _validate({})
    assert exc.value.detail["errors"][0]["type"] == "too_many_ladder_bands"


# ---------------------------------------------------------------- API 端到端

def _payload(**override):
    payload = dict(
        ladder_sizes=_sizes([10, 30, 55, 80, 110, 140]),
        ladder_distances=[10, 30, 55, 80, 110, 140],
        sample_distances=[20, 140, 5, 150],
        max_missing_or_spurious_bands=2,
        residual_threshold=0.01,
    )
    payload.update(override)
    return payload


def test_api_size_calibration_ok():
    res = client.post("/api/v1/digest/size-calibration", json=_payload())
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["anchor_count"] == 6
    assert body["skip_total"] == 0
    assert body["regression"]["within_threshold"] is True
    assert len(body["anchors"]) == 6
    assert body["missing_standard_bands"] == []
    assert body["spurious_bands"] == []
    statuses = [(b["distance"], b["status"]) for b in body["sample_bands"]]
    assert statuses == [
        (20, "calibrated"), (140, "calibrated"),
        (5, "out_of_range_above"), (150, "out_of_range_below"),
    ]
    assert body["sample_bands"][0]["estimated_size"] == pytest.approx(
        round(10 ** 2.8, 2))
    assert body["calibration_range_min_distance"] == 10
    assert body["calibration_range_max_distance"] == 140


def test_api_missing_and_spurious_bands_reported():
    # 缺第 3 条标准带（d=55）且多一条杂带（d=42）
    res = client.post("/api/v1/digest/size-calibration", json=_payload(
        ladder_distances=[10, 30, 42, 80, 110, 140],
        sample_distances=[55],
        max_missing_or_spurious_bands=2, residual_threshold=0.05))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["anchor_count"] == 5
    assert body["skip_total"] == 2
    assert [m["ladder_size_index"] for m in body["missing_standard_bands"]] == [3]
    assert [s["ladder_distance_index"] for s in body["spurious_bands"]] == [3]


def test_api_422_duplicate_distance():
    res = client.post("/api/v1/digest/size-calibration", json=_payload(
        ladder_distances=[10, 30, 30, 80, 110, 140]))
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert any(e["type"] == "duplicate_distance" for e in body["errors"])


def test_api_422_sizes_not_descending():
    res = client.post("/api/v1/digest/size-calibration", json=_payload(
        ladder_sizes=[1000, 500, 600, 100, 50, 20]))
    assert res.status_code == 422
    assert any(e["type"] == "ladder_sizes_not_descending"
               for e in res.json()["errors"])


def test_api_422_residual_threshold_exceeded():
    res = client.post("/api/v1/digest/size-calibration", json=_payload(
        ladder_sizes=[1000, 500, 100, 50, 20, 10],
        max_missing_or_spurious_bands=4, residual_threshold=0.001))
    assert res.status_code == 422
    assert any(e["type"] == "residual_threshold_exceeded"
               for e in res.json()["errors"])


def test_api_422_skip_budget_unreachable():
    res = client.post("/api/v1/digest/size-calibration", json=_payload(
        ladder_sizes=[1000, 500],
        ladder_distances=[10, 20, 30, 40],
        max_missing_or_spurious_bands=1, residual_threshold=1.0))
    assert res.status_code == 422
    assert any(e["type"] == "skip_limit_unreachable"
               for e in res.json()["errors"])


def test_api_422_field_constraints():
    # 残差阈值缺失、尺寸非正、上限为负、列表过短均由 Pydantic 拦截
    for bad in (
        {"residual_threshold": -0.01},
        {"ladder_sizes": [1000, -50]},
        {"max_missing_or_spurious_bands": -1},
        {"ladder_distances": [10]},
        {"ladder_sizes": [1000]},
    ):
        res = client.post("/api/v1/digest/size-calibration", json=_payload(**bad))
        assert res.status_code == 422, bad
        assert res.json()["code"] == "validation_failed"
