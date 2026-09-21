"""分子量标定（size-calibration）：锚点枚举、回归残差、分段插值与 422 拒绝的单元 + API 测试。"""
import math

from fastapi.testclient import TestClient

from app.main import app
from app.models import SizeCalibrationRequest
from app.size_calibration import calibrate_sizes

client = TestClient(app)


def _req(sizes, distances, samples=(), budget=0, threshold=0.02):
    return SizeCalibrationRequest(
        ladder_sizes=sizes,
        ladder_distances=distances,
        sample_distances=list(samples),
        max_missed_bands=budget,
        residual_threshold=threshold,
    )


# log10(bp) = 3.6 − 0.01·distance 的标准数据（近乎完美线性，保留两位距离）
SIZES = [2000, 1500, 1000, 750, 500, 250]
DISTANCES = [29.9, 42.39, 60.0, 72.49, 90.1, 120.21]


# ---------------------------------------------------------------- 基本标定

def test_clean_ladder_selects_all_anchors():
    res = calibrate_sizes(_req(SIZES, DISTANCES, samples=[45.0, 80.0]))
    assert res.anchor_count == 6
    assert [a.ladder_index for a in res.anchors] == [1, 2, 3, 4, 5, 6]
    # 锚点按距离递增 = 标准降序序号递增
    assert [a.distance for a in res.anchors] == sorted(DISTANCES)
    assert [a.size for a in res.anchors] == SIZES
    assert res.missing_ladder_count == 0
    assert res.spurious_ladder_count == 0
    # 近完美线性：斜率约 -0.01，残差极小
    assert abs(res.regression_slope + 0.01) < 1e-4
    assert res.residual_rmse < 0.001


def test_piecewise_linear_interpolation_and_no_extrapolation():
    res = calibrate_sizes(_req(
        [2000, 250], [30.0, 120.0], samples=[30.0, 75.0, 120.0, 29.0, 121.0],
        threshold=1.0,
    ))
    by_dist = {b.distance: b for b in res.sample_bands}
    # 端点距离取锚点本身
    assert by_dist[30.0].estimated_bp == 2000.0
    assert by_dist[120.0].estimated_bp == 250.0
    # log10(bp) 中点 -> 几何平均 sqrt(2000*250) ≈ 707.11
    mid = by_dist[75.0]
    assert mid.status == "estimated"
    assert abs(mid.estimated_bp - math.sqrt(2000 * 250)) < 0.01
    # 越界不外推
    low = by_dist[29.0]
    high = by_dist[121.0]
    assert (low.status, low.estimated_bp) == ("below_range", None)
    assert (high.status, high.estimated_bp) == ("above_range", None)
    assert res.estimated_count == 3
    assert res.out_of_range_count == 2


def test_sample_bands_sorted_ascending_independent_of_input_order():
    r1 = calibrate_sizes(_req(SIZES, DISTANCES, samples=[130.0, 20.0, 80.0, 45.0]))
    r2 = calibrate_sizes(_req(SIZES, DISTANCES, samples=[45.0, 20.0, 130.0, 80.0]))
    assert [b.distance for b in r1.sample_bands] == [20.0, 45.0, 80.0, 130.0]
    assert [b.model_dump() for b in r1.sample_bands] == [
        b.model_dump() for b in r2.sample_bands
    ]
    assert [b.band_index for b in r1.sample_bands] == [1, 2, 3, 4]


# ---------------------------------------------------------------- 杂带 / 缺带枚举

def test_spurious_ladder_band_is_skipped_and_residual_minimized():
    # 7 条实测：97.5 是插入在 500(90.1) 与 250(120.21) 之间的杂带
    distances = DISTANCES + [97.5]
    res = calibrate_sizes(_req(SIZES, distances, budget=1))
    assert res.anchor_count == 6
    assert res.missing_ladder_count == 0
    assert res.spurious_ladder_count == 1
    assert res.spurious_ladder_distances == [97.5]
    assert res.residual_rmse < 0.001


def test_same_skip_count_alignment_chosen_by_residual():
    # 3 条标准、4 条距离，跳过总数同为 1；95.0 离线最远，应被选为杂带
    res = calibrate_sizes(_req(
        [1000, 500, 250], [20.0, 35.0, 50.0, 95.0], budget=1, threshold=1.0,
    ))
    assert res.spurious_ladder_distances == [95.0]
    assert res.anchor_count == 3
    assert res.residual_rmse == 0.0


def test_missing_standard_band_is_reported_in_descending_order():
    # 5 条实测，缺少 2000bp 对应带
    distances = DISTANCES[1:]
    res = calibrate_sizes(_req(SIZES, distances, budget=2))
    assert res.anchor_count == 5
    assert [a.ladder_index for a in res.anchors] == [2, 3, 4, 5, 6]
    assert res.missing_ladder_sizes == [2000.0]
    assert res.spurious_ladder_distances == []


def test_tie_break_lexicographic_index_sequence():
    # 3 条标准、4 条距离，log10 等距，任一距离被跳过残差相同；
    # 字典序最优为跳过距离侧最大序号（保留近孔侧/大尺寸端锚点）
    sizes = [1000, 500, 250]
    distances = [20.0, 40.0, 60.0, 80.0]
    res = calibrate_sizes(_req(sizes, distances, budget=1, threshold=1.0))
    assert res.anchor_count == 3
    assert [a.distance for a in res.anchors] == [20.0, 40.0, 60.0]
    assert res.spurious_ladder_distances == [80.0]


def test_ladder_distances_may_be_submitted_out_of_order():
    res1 = calibrate_sizes(_req(SIZES, DISTANCES, samples=[45.0]))
    shuffled = list(reversed(DISTANCES))
    res2 = calibrate_sizes(_req(SIZES, shuffled, samples=[45.0]))
    assert res1.model_dump() == res2.model_dump()


def test_sample_distance_may_equal_anchor_distance():
    res = calibrate_sizes(_req(SIZES, DISTANCES, samples=[60.0]))
    band = res.sample_bands[0]
    assert band.status == "estimated"
    assert band.estimated_bp == 1000.0


# ---------------------------------------------------------------- 422 拒绝

def _post(payload):
    return client.post("/api/v1/digest/size-calibration", json=payload)


def _expect_422(payload, code_part=None):
    res = _post(payload)
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert body["errors"]
    if code_part:
        assert any(code_part in e["type"] for e in body["errors"])
    return body


def test_422_duplicate_ladder_distance():
    _expect_422({
        "ladder_sizes": SIZES,
        "ladder_distances": [29.9, 29.9, 60.0, 72.49, 90.1, 120.21],
        "sample_distances": [], "max_missed_bands": 0, "residual_threshold": 0.02,
    }, "duplicate_migration_distance")


def test_422_duplicate_sample_distance():
    _expect_422({
        "ladder_sizes": SIZES,
        "ladder_distances": DISTANCES,
        "sample_distances": [45.0, 45.0],
        "max_missed_bands": 0, "residual_threshold": 0.02,
    }, "duplicate_migration_distance")


def test_422_ladder_sizes_not_descending():
    body = _expect_422({
        "ladder_sizes": [2000, 1500, 1500, 750, 500, 250],
        "ladder_distances": DISTANCES,
        "sample_distances": [], "max_missed_bands": 0, "residual_threshold": 0.02,
    }, "ladder_sizes_not_descending")
    assert any("ladder_sizes" in e["loc"] for e in body["errors"])

    _expect_422({
        "ladder_sizes": [1500, 2000, 1000, 750, 500, 250],
        "ladder_distances": DISTANCES,
        "sample_distances": [], "max_missed_bands": 0, "residual_threshold": 0.02,
    }, "ladder_sizes_not_descending")


def test_422_insufficient_anchors_when_skip_budget_too_small():
    # 6 标准 vs 7 实测，至少需要 1 次跳过
    body = _expect_422({
        "ladder_sizes": SIZES,
        "ladder_distances": DISTANCES + [97.5],
        "sample_distances": [], "max_missed_bands": 0, "residual_threshold": 0.02,
    }, "insufficient_calibration_anchors")
    assert any("max_missed_bands" in e["msg"] for e in body["errors"])


def test_422_when_ladder_lists_shorter_than_two():
    # 两侧 ladder 列表均要求至少 2 条（否则不可能取得两个锚点），由字段约束直接拦截
    res = _post({
        "ladder_sizes": [2000, 1000, 250],
        "ladder_distances": [60.0],
        "sample_distances": [], "max_missed_bands": 5, "residual_threshold": 1.0,
    })
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert any("ladder_distances" in str(e["loc"]) for e in body["errors"])


def test_422_residual_threshold_exceeded():
    # 完美线性 ladder 的 RMSE≈4.3e-5，给极小阈值触发拒绝
    body = _expect_422({
        "ladder_sizes": SIZES,
        "ladder_distances": DISTANCES,
        "sample_distances": [], "max_missed_bands": 0, "residual_threshold": 0.000001,
    }, "residual_threshold_exceeded")
    assert any("residual_threshold" in e["loc"] for e in body["errors"])


def test_422_basic_field_constraints():
    base = {
        "ladder_sizes": SIZES, "ladder_distances": DISTANCES,
        "sample_distances": [], "max_missed_bands": 0, "residual_threshold": 0.02,
    }
    # 标准尺寸非正
    bad = dict(base, ladder_sizes=[2000, -250])
    assert _post(bad).status_code == 422
    # 负距离
    bad = dict(base, ladder_distances=[30.0, -1.0])
    assert _post(bad).status_code == 422
    # 负缺带上限 / 负阈值
    bad = dict(base, max_missed_bands=-1)
    assert _post(bad).status_code == 422
    bad = dict(base, residual_threshold=-0.1)
    assert _post(bad).status_code == 422
    # 列表不足 2 条
    bad = dict(base, ladder_sizes=[2000])
    assert _post(bad).status_code == 422
    # 缺字段
    assert _post({"ladder_sizes": SIZES}).status_code == 422


# ---------------------------------------------------------------- API 端到端

def test_api_endpoint_with_example_file():
    import json
    payload = json.load(open("examples/size-calibration-request.json"))
    res = _post(payload)
    assert res.status_code == 200
    body = res.json()
    assert body["anchor_count"] == 6
    assert body["spurious_ladder_distances"] == [97.5]
    assert body["missing_ladder_sizes"] == []
    statuses = [(b["distance"], b["status"], b["estimated_bp"]) for b in body["sample_bands"]]
    assert statuses == [
        (20.0, "below_range", None),
        (45.0, "estimated", 1412.51),
        (80.0, "estimated", 630.91),
        (130.0, "above_range", None),
    ]
    assert body["calibration_reason"]
    assert body["residual_rmse"] <= body["residual_threshold"]


def test_api_order_independence():
    import json
    payload = json.load(open("examples/size-calibration-request.json"))
    shuffled = dict(payload)
    shuffled["ladder_distances"] = list(reversed(payload["ladder_distances"]))
    shuffled["sample_distances"] = list(reversed(payload["sample_distances"]))
    assert _post(payload).json() == _post(shuffled).json()
