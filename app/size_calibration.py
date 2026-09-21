"""核心算法：凝胶迁移距离到片段 bp 的分子量校准（标准 ladder 锚点对齐 + 分段插值）。

所有函数均为纯函数，不依赖任何外部服务或全局状态。

锚点对齐
--------
请求中的 ``ladder_sizes`` 按片段尺寸严格降序提交（标准 ladder 在凝胶上
尺寸越大迁移越慢、距离越小）；``ladder_distances`` 为 ladder 泳道的实测
条带距离，允许相对标准 ladder 存在缺带（标准带未在实测中出现）与杂带
（实测条带不属于 ladder）。

一对一对齐在"迁移距离递增、log10(bp) 递减"的顺序约束下枚举：标准带按
尺寸降序对应 log10(bp) 降序，实测带按距离升序排列后，任一可行对齐都是
两组严格递增下标序列 ``i_1 < i_2 < … < i_k``（实测，按距离升序的序号）与
``j_1 < j_2 < … < j_k``（标准，按尺寸降序的序号），即一个保序的一对一
锚点匹配；未入选实测带为杂带、未入选标准带为缺带，允许跳过两侧项目。

优选次序
--------
全部可行对齐依次按：

1. **跳过总数最小**（缺带数 + 杂带数，等价于锚点对数最多）；
2. **线性回归残差最小**：入选锚点在 (距离, log10(bp)) 空间做最小二乘
   线性拟合，比较 log10(bp) 残差的均方根（RMSE）；
3. **索引序列字典序最小**：比较锚点的 (实测下标, 标准下标) 配对序列。

第 1 级把枚举规模限制在单一锚点层数 k（由两侧长度与跳过上限决定），
第 3 级保证即使前两级完全并列也有唯一、确定的结果。

分段插值与禁外推
----------------
入选锚点按距离升序构成分段线性插值函数 ``log10(bp) = f(距离)``，
待测带距离落在相邻锚点之间时线性插值后取 10^log10(bp)；

* 距离 < 最小锚点距离（``out_of_range_above``，片段比最大锚点还大）；
* 距离 > 最大锚点距离（``out_of_range_below``，片段比最小锚点还小）；

均**不外推**，返回越界状态且不给出 bp 估算。恰好等于某锚点距离时直接
取该锚点 bp。
"""
import math
from itertools import combinations

from app.models import (
    CalibratedSampleBandOut,
    MissingStandardBandOut,
    SizeAnchorOut,
    SizeCalibrationRequest,
    SizeCalibrationResponse,
    SizeRegressionOut,
    SpuriousBandOut,
)
from app.validation import DigestValidationError, _err

# RMSE 视为并列的浮点容差（log10(bp) 单位）：差异小于该值视为残差相同，
# 改由第三级索引序列字典序破同分，避免浮点噪声改变确定性结果
_RMSE_TIE_EPSILON = 1e-12


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """对 (xs, ys) 做最小二乘线性拟合 y = slope·x + intercept。

    返回 (slope, intercept, rmse, r_squared)；xs 互不相同且点数 ≥ 2 时
    Sxx 恒为正。调用方保证锚点距离互不相同、锚点至少 2 个。
    """
    k = len(xs)
    mean_x = sum(xs) / k
    mean_y = sum(ys) / k
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    sse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    if sse < 0:  # 浮点误差保护
        sse = 0.0
    rmse = math.sqrt(sse / k)
    syy = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 - sse / syy if syy > 0 else 1.0
    return slope, intercept, rmse, r_squared


def calibrate_sizes(req: SizeCalibrationRequest) -> SizeCalibrationResponse:
    """枚举最优锚点对齐，回归校验并对每条待测带做禁外推分段插值。"""
    # 标准 ladder：按请求的尺寸降序（log10(bp) 降序）；下标 0 对应最大尺寸
    std_sizes = [float(s) for s in req.ladder_sizes]
    std_logs = [math.log10(s) for s in std_sizes]
    n = len(std_sizes)

    # 实测 ladder：按距离升序排序（提交序号单独保留用于报告）
    measured = sorted(
        (float(d), idx) for idx, d in enumerate(req.ladder_distances)
    )
    meas_dists = [d for d, _ in measured]
    m = len(meas_dists)

    max_skips = req.max_missing_or_spurious_bands

    # 一级目标：跳过总数 = (n - k) + (m - k) 最小 → 锚点数 k 最大。
    # 跳过总数的理论最小值为 |n - m|（全部跳过长出的一侧），对应
    # k = min(n, m)。该最小值仍超过上限时，跳过预算内不存在任何对齐。
    if abs(n - m) > max_skips:
        raise DigestValidationError([_err(
            ["max_missing_or_spurious_bands"],
            f"标准 ladder {n} 条、实测 ladder {m} 条，即使全部锚点对齐仍至少"
            f"跳过 {abs(n - m)} 项（缺带 + 杂带），超过允许上限 {max_skips}，"
            "跳过预算内不存在满足顺序约束的一对一锚点对齐",
            "skip_limit_unreachable",
        )])
    k = min(n, m)
    if k < 2:
        raise DigestValidationError([_err(
            ["ladder_sizes"],
            f"标准 ladder {n} 条、实测 ladder {m} 条，至多可形成 {k} 个"
            "一对一锚点，无法取得至少 2 个锚点，不能进行距离-bp 校准",
            "insufficient_anchors",
        )])

    # 在固定锚点层数 k 上枚举全部保序对齐：
    # I 为实测（按距离升序）下标，J 为标准（按尺寸降序）下标，
    # 配对序列 (I_t, J_t) 即"距离递增、log10(bp) 递减"的一对一锚点。
    # combinations 双层枚举顺序即配对序列的字典序（先比实测下标、再比标准下标），
    # 故仅当 RMSE 严格更小（超过浮点并列容差）才替换，并列时天然保留字典序最小者。
    best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    for ii in combinations(range(m), k):
        for jj in combinations(range(n), k):
            xs = [meas_dists[i] for i in ii]
            ys = [std_logs[j] for j in jj]
            _, _, rmse, _ = _linear_regression(xs, ys)
            if best is None or rmse < best[0] - _RMSE_TIE_EPSILON:
                best = (rmse, ii, jj)
    assert best is not None
    rmse, ii, jj = best

    # 残差阈值：超出则整体拒绝（422）
    if rmse > float(req.residual_threshold):
        raise DigestValidationError([_err(
            ["residual_threshold"],
            f"最优锚点对齐（{k} 个锚点，跳过 {n + m - 2 * k} 项）在"
            f"距离-log10(bp) 空间的线性回归 RMSE 为 {rmse:.6f} log10(bp)，"
            f"超过残差阈值 {float(req.residual_threshold):g}，"
            "ladder 实测距离与标准尺寸的线性关系不达标，拒绝校准",
            "residual_threshold_exceeded",
        )])

    xs = [meas_dists[i] for i in ii]
    ys = [std_logs[j] for j in jj]
    slope, intercept, _, r_squared = _linear_regression(xs, ys)

    used_meas = set(ii)
    used_std = set(jj)

    # 锚点报告：按迁移距离升序（枚举顺序即距离升序）
    anchors: list[SizeAnchorOut] = []
    for anchor_no, (i, j) in enumerate(zip(ii, jj), start=1):
        dist = meas_dists[i]
        size = std_sizes[j]
        fitted = slope * dist + intercept
        anchors.append(SizeAnchorOut(
            anchor_index=anchor_no,
            ladder_size=size,
            ladder_size_index=j + 1,
            ladder_distance=dist,
            ladder_distance_index=measured[i][1] + 1,
            fitted_log10_size=round(fitted, 6),
            residual_log10=round(std_logs[j] - fitted, 6),
            description=(
                f"锚点 {anchor_no}：实测迁移距离 {dist:g}（提交序号 "
                f"{measured[i][1] + 1}）与标准带 {size:g} bp"
                f"（ladder_sizes 降序序号 {j + 1}）一一对应，"
                f"log10(bp)={std_logs[j]:.6f}，拟合值 {fitted:.6f}"
            ),
        ))

    # 缺失标准带：未入选锚点的标准项，按 ladder_sizes 序号（尺寸降序）升序
    missing: list[MissingStandardBandOut] = [
        MissingStandardBandOut(
            ladder_size_index=j + 1,
            ladder_size=std_sizes[j],
            description=(
                f"标准带 {std_sizes[j]:g} bp（ladder_sizes 降序序号 {j + 1}）"
                "未在 ladder 实测距离中找到保序对应，计为缺失标准带"
            ),
        )
        for j in sorted(used_std ^ set(range(n)))
    ]

    # 杂带：未入选锚点的实测项，按迁移距离升序
    spurious: list[SpuriousBandOut] = [
        SpuriousBandOut(
            ladder_distance_index=measured[i][1] + 1,
            distance=meas_dists[i],
            description=(
                f"实测迁移距离 {meas_dists[i]:g}（提交序号 "
                f"{measured[i][1] + 1}）未与任何标准 ladder 条带保序对应，"
                "计为杂带"
            ),
        )
        for i in sorted(used_meas ^ set(range(m)))
    ]

    regression = SizeRegressionOut(
        slope=slope,
        intercept=intercept,
        rmse=round(rmse, 6),
        r_squared=round(r_squared, 6),
        residual_threshold=float(req.residual_threshold),
        within_threshold=rmse <= float(req.residual_threshold),
    )

    # 可校准范围：锚点距离的最小 / 最大值（锚点按距离升序）
    min_dist, max_dist = xs[0], xs[-1]
    max_size, min_size = 10 ** ys[0], 10 ** ys[-1]

    # 待测带：在相邻锚点间分段线性插值，超出锚点范围禁止外推
    sample_results: list[CalibratedSampleBandOut] = []
    for submit_idx, raw_dist in enumerate(req.sample_distances):
        dist = float(raw_dist)
        if dist < min_dist:
            sample_results.append(CalibratedSampleBandOut(
                sample_band_index=submit_idx + 1,
                distance=dist,
                status="out_of_range_above",
                estimated_size=None,
                segment_anchor_indexes=None,
                description=(
                    f"待测带距离 {dist:g} 小于最小锚点距离 {min_dist:g}"
                    f"（对应最大锚点 {max_size:g} bp），片段比最大锚点还大；"
                    "超出锚点范围禁止外推，不给出 bp 估算"
                ),
            ))
            continue
        if dist > max_dist:
            sample_results.append(CalibratedSampleBandOut(
                sample_band_index=submit_idx + 1,
                distance=dist,
                status="out_of_range_below",
                estimated_size=None,
                segment_anchor_indexes=None,
                description=(
                    f"待测带距离 {dist:g} 大于最大锚点距离 {max_dist:g}"
                    f"（对应最小锚点 {min_size:g} bp），片段比最小锚点还小；"
                    "超出锚点范围禁止外推，不给出 bp 估算"
                ),
            ))
            continue

        # 落在 [min_dist, max_dist]：定位所在锚点（xs 严格升序）
        seg = 0
        while seg + 1 < k and xs[seg + 1] <= dist:
            seg += 1
        if dist == xs[seg]:
            log_bp = ys[seg]
            segment = [seg + 1]
            seg_desc = f"恰好落在锚点 {seg + 1}（{10 ** ys[seg]:g} bp）上"
        else:
            x0, x1 = xs[seg], xs[seg + 1]
            y0, y1 = ys[seg], ys[seg + 1]
            log_bp = y0 + (y1 - y0) * (dist - x0) / (x1 - x0)
            segment = [seg + 1, seg + 2]
            seg_desc = (
                f"在锚点 {seg + 1}（{x0:g} → {10 ** y0:g} bp）与锚点 "
                f"{seg + 2}（{x1:g} → {10 ** y1:g} bp）之间对 log10(bp) "
                "分段线性插值"
            )

        estimated = round(10 ** log_bp, 2)
        sample_results.append(CalibratedSampleBandOut(
            sample_band_index=submit_idx + 1,
            distance=dist,
            status="calibrated",
            estimated_size=estimated,
            segment_anchor_indexes=segment,
            description=(
                f"待测带距离 {dist:g}：{seg_desc}，得到 log10(bp)={log_bp:.6f}，"
                f"估算片段尺寸约 {estimated:g} bp"
            ),
        ))

    reason = (
        f"标准 ladder {n} 条（尺寸严格降序）与 ladder 实测 {m} 条"
        f"（按迁移距离升序）在“距离递增、log10(bp) 递减”的顺序约束下枚举"
        "一对一锚点对齐（未配对的标准带计缺带、未配对的实测带计杂带，两侧"
        "均可跳过）；在跳过总数不超过 "
        f"{max_skips} 的全部对齐中依次以 ① 跳过总数最小（"
        f"{n + m - 2 * k} = 缺带 {n - k} + 杂带 {m - k}，锚点 {k} 个）、"
        "② 锚点距离-log10(bp) 最小二乘回归 RMSE 最小"
        f"（{rmse:.6f} log10(bp)，阈值 {float(req.residual_threshold):g}）、"
        "③ (实测下标, 标准下标) 索引序列字典序最小选定唯一对齐；"
        "待测带在相邻锚点间对 log10(bp) 分段线性插值后取 10^log10(bp)，"
        f"可校准距离范围为 [{min_dist:g}, {max_dist:g}]"
        f"（约 {max_size:g} ~ {min_size:g} bp），超出锚点范围一律不外推"
    )

    return SizeCalibrationResponse(
        ladder_band_count=n,
        measured_band_count=m,
        sample_band_count=len(req.sample_distances),
        max_missing_or_spurious_bands=max_skips,
        anchor_count=k,
        missing_standard_count=n - k,
        spurious_band_count=m - k,
        skip_total=n + m - 2 * k,
        anchors=anchors,
        missing_standard_bands=missing,
        spurious_bands=spurious,
        regression=regression,
        calibration_range_min_distance=min_dist,
        calibration_range_max_distance=max_dist,
        calibration_range_min_size=round(min_size, 2),
        calibration_range_max_size=round(max_size, 2),
        sample_bands=sample_results,
        calibration_reason=reason,
    )
