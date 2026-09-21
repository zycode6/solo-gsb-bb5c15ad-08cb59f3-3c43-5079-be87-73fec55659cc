"""核心算法：以标准 ladder 把凝胶迁移距离标定为片段 bp（分子量标定）。

所有函数均为纯函数，不依赖任何外部服务或全局状态。

锚点对齐枚举
------------
标准 ladder 尺寸按**降序**给出（大段在前），其 ``log10(bp)`` 严格递减；
实测 ladder 迁移距离按**递增**排列（离加样孔越远迁移越多）。一次合法的
一对一锚点对齐就是两条序列间的一组**保序配对**：标准侧选中的降序序号
严格递增、距离侧选中的升序序号也严格递增（LCS 式对齐），允许跳过两侧的
项目——未被选中的标准带为**缺失标准带**，未被选中的实测 ladder 带为
**杂带**。对齐在几何上即满足“迁移距离递增、log10(bp) 递减”的单调约束。

三级优化目标（按次序，字典序破同分保证唯一结果）：

  1. **跳过总数最少**：``缺失标准带 + 杂带 = n + m − 2·锚点数`` 最小，
     即锚点数 ``k`` 取最大可行值 ``min(n, m)``（锚点更少的对齐跳过更多，
     不可能更优）；跳过总数超过 ``max_missed_bands`` 时不存在可行对齐；
  2. **线性回归残差最小**：在入选锚点的 ``(distance, log10(size))`` 上
     最小二乘拟合 ``log10(bp) = a + b·distance``，取锚点残差
     ``RMSE = sqrt(Σ(y − ŷ)² / k)`` 最小的对齐；
  3. **索引序列字典序最小**：残差并列时，取按距离递增排列的
     ``(标准降序序号, 距离升序序号)`` 索引对序列字典序最小者
     （等效优先保留靠近大尺寸端 / 近孔侧的带）。

最大可行锚点数小于 2，或最小跳过总数已超过 ``max_missed_bands`` 时，
无法取得至少两个锚点，抛出 :class:`CalibrationError`（HTTP 422）。

分段线性插值（绝不外推）
------------------------
以入选锚点在距离-log10(bp) 空间做**分段线性插值**：待测距离落在
``[最小锚点距离, 最大锚点距离]`` 内时，在所在锚点区间内线性插值得到
``log10(bp)`` 后取 ``10 ** log10(bp)``；严格小于最小锚点距离
（``below_range``）或严格大于最大锚点距离（``above_range``）一律
**不外推**，只返回越界状态。

最优对齐的回归 RMSE 超过请求残差阈值时抛出 :class:`CalibrationError`
（HTTP 422）；阈值比较使用未舍入的原始残差。
"""
import math

from app.models import (
    CalibrationAnchorOut,
    CalibrationSampleBandOut,
    SizeCalibrationRequest,
    SizeCalibrationResponse,
)

# 并列残差视为相等的相对容差：小于该差的 RMSE 差异不再区分，转由索引序列破同分
_RESIDUAL_EPS = 1e-12


class CalibrationError(Exception):
    """算法级标定失败：无法取得至少两个锚点或残差超阈值，统一映射为 HTTP 422。"""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__(str(errors))


def _regression(stats: tuple[int, float, float, float, float]):
    """由充分统计量 (k, Σx, Σy, Σx², Σxy) 求最小二乘 (截距, 斜率)。"""
    k, sx, sy, sxx, sxy = stats
    mean_x = sx / k
    mean_y = sy / k
    var_x = sxx - sx * sx / k
    cov_xy = sxy - sx * sy / k
    # 锚点距离两两不同（实测 ladder 距离不可重复），var_x 恒为正
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _enumerate_best_alignment(
    log_sizes: list[float],
    distances: list[float],
):
    """枚举跳过最少（锚点数 k = min(n,m)）的全部保序对齐，返回最优锚点。

    锚点数固定为 k 时跳过总数恒为 n+m−2k，已是最小值；锚点更少的对齐
    跳过更多，无需枚举。枚举顺序即为索引对序列的字典序：标准侧序号
    元组优先、距离侧序号元组其次，因此首个达到当前最优残差的对齐就是
    字典序最小的并列最优解。
    """
    n = len(log_sizes)
    m = len(distances)
    k = min(n, m)

    best_indices: list[tuple[int, int]] | None = None
    best_rmse = math.inf

    def evaluate(pairs: list[tuple[int, int]],
                 stats: tuple[int, float, float, float, float]) -> None:
        nonlocal best_indices, best_rmse
        kk, _, _, _, _ = stats
        intercept, slope = _regression(stats)
        sse = 0.0
        for si, di in pairs:
            residual = log_sizes[si] - (intercept + slope * distances[di])
            sse += residual * residual
        rmse = math.sqrt(sse / kk)
        if rmse < best_rmse - _RESIDUAL_EPS:
            best_rmse = rmse
            best_indices = list(pairs)

    # 深度优先、按下标递增的顺序枚举 k 个保序配对；充分统计量随路径增量累加
    def dfs(t: int, si: int, di: int,
            pairs: list[tuple[int, int]],
            sx: float, sy: float, sxx: float, sxy: float) -> None:
        if t == k:
            evaluate(pairs, (k, sx, sy, sxx, sxy))
            return
        # 还需选 remain 个配对：下一标准下标范围为 si+1 .. n-remain，
        # 下一距离下标范围为 di+1 .. m-remain（两侧各自预留，互不耦合）
        remain = k - t
        for next_si in range(si + 1, n - remain + 1):
            yv = log_sizes[next_si]
            for next_di in range(di + 1, m - remain + 1):
                xv = distances[next_di]
                pairs.append((next_si, next_di))
                dfs(t + 1, next_si, next_di,
                    pairs, sx + xv, sy + yv, sxx + xv * xv, sxy + xv * yv)
                pairs.pop()

    dfs(0, -1, -1, [], 0.0, 0.0, 0.0, 0.0)
    return k, best_indices, best_rmse


def calibrate_sizes(req: SizeCalibrationRequest) -> SizeCalibrationResponse:
    """枚举最优 ladder 锚点对齐、做回归质量校验并分段线性插值估算待测带 bp。"""
    # 标准尺寸降序给出 -> log10 严格递减；实测距离排序后严格递增（重复已在语义校验拦截）
    sizes = [float(s) for s in req.ladder_sizes]
    log_sizes = [math.log10(s) for s in sizes]
    ordered_distances = sorted(float(d) for d in req.ladder_distances)
    sample_distances = sorted(float(d) for d in req.sample_distances)

    n = len(sizes)
    m = len(ordered_distances)
    max_anchors = min(n, m)
    min_skips = n + m - 2 * max_anchors

    if max_anchors < 2 or min_skips > req.max_missed_bands:
        raise CalibrationError([{
            "loc": ["ladder_sizes"],
            "msg": (
                f"无法取得至少两个锚点：标准 ladder 带 {n} 条、实测 ladder 带 {m} 条，"
                f"最多 {max_anchors} 个锚点，最少跳过 {min_skips} 项"
                f"（缺失标准带 + 杂带），而 max_missed_bands={req.max_missed_bands}"
            ),
            "type": "insufficient_calibration_anchors",
        }])

    k, best_pairs, best_rmse = _enumerate_best_alignment(log_sizes, ordered_distances)

    if best_rmse > float(req.residual_threshold) + _RESIDUAL_EPS:
        raise CalibrationError([{
            "loc": ["residual_threshold"],
            "msg": (
                f"最优锚点对齐（{k} 个锚点）的线性回归残差 RMSE={best_rmse:.6f}"
                f"（log10(bp) 量纲）超过残差阈值 {float(req.residual_threshold):g}，"
                "标定直线不可用"
            ),
            "type": "residual_threshold_exceeded",
        }])

    chosen_si = {si for si, _ in best_pairs}
    chosen_di = {di for _, di in best_pairs}

    anchors: list[CalibrationAnchorOut] = [
        CalibrationAnchorOut(
            ladder_index=si + 1,
            size=sizes[si],
            distance=ordered_distances[di],
            log10_size=round(log_sizes[si], 6),
        )
        for si, di in best_pairs
    ]

    # 缺失标准带：标准侧未入选的尺寸，按降序输入序号（即尺寸降序）输出
    missing_sizes = [sizes[si] for si in range(n) if si not in chosen_si]
    # 杂带：距离侧未入选的实测 ladder 距离，按距离递增输出
    spurious_distances = [
        ordered_distances[di] for di in range(m) if di not in chosen_di
    ]

    # 由锚点重新汇总充分统计量，给出回归参数（与枚举内部一致）
    xs = [a.distance for a in anchors]
    ys = [log_sizes[si] for si, _ in best_pairs]
    kk = len(anchors)
    mean_x = sum(xs) / kk
    mean_y = sum(ys) / kk
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
    intercept = mean_y - slope * mean_x

    # 分段线性插值：仅在锚点跨度 [d_min, d_max] 内估算，越界不外推
    anchor_x = xs
    anchor_y = ys
    d_min, d_max = anchor_x[0], anchor_x[-1]
    sample_bands: list[CalibrationSampleBandOut] = []
    estimated = 0
    out_of_range = 0
    for band_no, dist in enumerate(sample_distances, start=1):
        if dist < d_min:
            status = "below_range"
            bp: float | None = None
            out_of_range += 1
            desc = (
                f"待测带 {band_no} 迁移距离 {dist:g} 小于最小锚点距离 {d_min:g}"
                f"（近孔侧，对应 bp 大于最大锚点 {10 ** anchor_y[0]:g}），超出锚点范围不外推"
            )
        elif dist > d_max:
            status = "above_range"
            bp = None
            out_of_range += 1
            desc = (
                f"待测带 {band_no} 迁移距离 {dist:g} 大于最大锚点距离 {d_max:g}"
                f"（远孔侧，对应 bp 小于最小锚点 {10 ** anchor_y[-1]:g}），超出锚点范围不外推"
            )
        else:
            # 定位所在锚点区间（相邻锚点距离严格递增），线性插值 log10(bp)
            seg = 0
            while seg + 1 < kk and anchor_x[seg + 1] < dist:
                seg += 1
            x0, x1 = anchor_x[seg], anchor_x[seg + 1]
            y0, y1 = anchor_y[seg], anchor_y[seg + 1]
            log_bp = y0 if x1 == x0 else y0 + (dist - x0) * (y1 - y0) / (x1 - x0)
            bp = round(10 ** log_bp, 2)
            status = "estimated"
            estimated += 1
            desc = (
                f"待测带 {band_no} 迁移距离 {dist:g} 落在锚点 {x0:g}（{10 ** y0:g} bp）"
                f"与 {x1:g}（{10 ** y1:g} bp）之间，分段线性插值 log10(bp)={log_bp:.6f}，"
                f"估算 {bp:g} bp"
            )
        sample_bands.append(CalibrationSampleBandOut(
            band_index=band_no,
            distance=dist,
            status=status,
            estimated_bp=bp,
            description=desc,
        ))

    reason = (
        f"标准 ladder {n} 条降序尺寸与 {m} 条实测迁移距离按"
        "“距离递增、log10(bp) 递减”的保序约束枚举一对一锚点对齐，允许跳过两侧项目"
        f"（缺带/杂带总数上限 {req.max_missed_bands:g}）；"
        f"最优对齐含 {k} 个锚点，跳过 {min_skips} 项"
        f"（缺失标准带 {len(missing_sizes)} 条、杂带 {len(spurious_distances)} 条），"
        "为跳过总数最少的对齐；同级中取距离-log10(bp) 空间最小二乘回归残差 RMSE "
        f"{best_rmse:.6f}（阈值 {float(req.residual_threshold):g}）最小者，"
        "残差并列时取索引对序列字典序最小者；"
        f"待测带在锚点距离区间 [{d_min:g}, {d_max:g}] 内按分段线性插值估算 bp，"
        "区间外一律不外推（below_range / above_range）"
    )

    return SizeCalibrationResponse(
        anchor_count=k,
        anchors=anchors,
        missing_ladder_count=len(missing_sizes),
        missing_ladder_sizes=missing_sizes,
        spurious_ladder_count=len(spurious_distances),
        spurious_ladder_distances=spurious_distances,
        regression_intercept=round(intercept, 6),
        regression_slope=round(slope, 6),
        residual_rmse=round(best_rmse, 6),
        residual_threshold=float(req.residual_threshold),
        sample_band_count=len(sample_distances),
        estimated_count=estimated,
        out_of_range_count=out_of_range,
        sample_bands=sample_bands,
        calibration_reason=reason,
    )
