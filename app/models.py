"""请求 / 响应数据模型。

输入使用 Pydantic 做基础类型约束；跨字段的语义校验
（位点越界、重复位点、切点数量等）在 :mod:`app.validation` 中完成，
两者最终都输出统一结构的 422 响应。
"""
from typing import Annotated, Literal

from pydantic import BaseModel, Field

SampleName = Literal["reference", "variant"]

# 条带长度 / 匹配容差：非负有限浮点数（JSON 整数自动转换）
NonNegativeBand = Annotated[float, Field(ge=0, allow_inf_nan=False)]

# 片段尺寸等必须严格为正的有限浮点数
PositiveBand = Annotated[float, Field(gt=0, allow_inf_nan=False)]


# ---------------------------------------------------------------- 输入模型

class EnzymeInput(BaseModel):
    name: str = Field(..., min_length=1, description="酶名称，如 EcoRI")
    reference_sites: list[int] = Field(
        default_factory=list, description="参考型上该酶的候选切点位置（bp，0-based）"
    )
    variant_sites: list[int] = Field(
        default_factory=list, description="变异型上该酶的候选切点位置（bp，0-based）"
    )


class DigestRequest(BaseModel):
    plasmid_length: int = Field(..., gt=0, description="环状质粒总长度（bp）")
    min_visible_fragment: int = Field(
        ..., ge=0, description="最小可见片段长度（bp），短于此值的片段不参与条带聚类"
    )
    band_resolution_tolerance: int = Field(
        ..., ge=0, description="条带分辨容差（bp），长度差不超过该值的相邻片段聚为同一条带"
    )
    enzymes: list[EnzymeInput] = Field(..., min_length=2, description="参与枚举的酶，至少 2 种")


# ---------------------------------------------------------------- 输出模型

class CutPointOut(BaseModel):
    position: int
    enzymes: list[str] = Field(description="在该位置产生切点的酶（可能两酶同位点）")


class FragmentOut(BaseModel):
    start: int = Field(description="片段起始切点位置")
    end: int = Field(description="片段终止切点位置（跨原点时为环绕后的位置）")
    length: int
    crosses_origin: bool = Field(description="是否为跨过坐标原点（0 点）的片段")


class SampleDigestOut(BaseModel):
    cuts: list[CutPointOut]
    fragments: list[FragmentOut]
    visible_fragment_count: int
    hidden_fragment_count: int = Field(description="短于最小可见片段、未参与成带的片段数")


class BandMemberOut(FragmentOut):
    sample: SampleName


class BandOut(BaseModel):
    band_id: int = Field(description="条带编号，按代表长度升序，从 1 开始")
    representative_length: float = Field(description="条带内片段长度平均值（bp）")
    min_length: int
    max_length: int
    reference_count: int
    variant_count: int
    distinguishable: bool = Field(description="两种样本在该条带中的片段数是否不同")
    members: list[BandMemberOut]


class DifferenceOut(BaseModel):
    band_id: int
    representative_length: float
    difference_type: Literal["reference_only", "variant_only", "count_mismatch"]
    reference_count: int
    variant_count: int
    description: str = Field(description="中文差异依据说明")


class PairRecommendationOut(BaseModel):
    enzyme_pair: list[str]
    pair_key: str
    samples: dict[SampleName, SampleDigestOut]
    bands: list[BandOut]
    total_bands: int
    distinguishable_band_count: int
    differences: list[DifferenceOut]


class RejectedPairOut(BaseModel):
    enzyme_pair: list[str]
    pair_key: str
    reason_code: Literal["insufficient_cuts"]
    reason: str
    reference_cut_count: int
    variant_cut_count: int


class NonDiscriminatingPairOut(BaseModel):
    enzyme_pair: list[str]
    pair_key: str
    total_bands: int = Field(description="两种样本条带完全一致时的条带总数")


class DigestResponse(BaseModel):
    plasmid_length: int
    min_visible_fragment: int
    band_resolution_tolerance: int
    total_enzyme_pairs: int
    recommended_count: int
    rejected_pair_count: int
    non_discriminating_pair_count: int
    recommendations: list[PairRecommendationOut]
    rejected_pairs: list[RejectedPairOut]
    non_discriminating_pairs: list[NonDiscriminatingPairOut]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


# ---------------------------------------------------------------- 鉴别（identify）模型

class IdentifyRequest(BaseModel):
    reference_bands: list[NonNegativeBand] = Field(
        default_factory=list, description="参考型预测条带长度列表（bp）"
    )
    variant_bands: list[NonNegativeBand] = Field(
        default_factory=list, description="变异型预测条带长度列表（bp）"
    )
    observed_bands: list[NonNegativeBand] = Field(
        ..., min_length=1, description="一次凝胶观测到的条带长度列表（bp），至少 1 条"
    )
    band_match_tolerance: NonNegativeBand = Field(
        ..., description="条带匹配容差（bp）：长度差不超过该值的预测/观测条带才允许配对"
    )


class BandMatchOut(BaseModel):
    predicted_length: float = Field(description="预测条带长度（bp）")
    observed_length: float = Field(description="与之配对的观测条带长度（bp）")
    error: float = Field(description="预测与观测长度差的绝对值（bp）")
    description: str = Field(description="中文匹配依据说明")


class SampleMatchOut(BaseModel):
    predicted_band_count: int
    matched_count: int
    missing_count: int = Field(description="缺失条带数：未在观测中找到配对的预测条带")
    extra_count: int = Field(description="额外条带数：未匹配到任何预测条带的观测条带")
    total_error: float = Field(description="全部匹配对的长度差之和（bp）")
    matches: list[BandMatchOut] = Field(description="逐条匹配依据，按预测长度升序")
    missing_bands: list[float] = Field(description="缺失条带：未匹配到观测的预测长度，升序")
    extra_bands: list[float] = Field(description="额外条带：未匹配到预测的观测长度，升序")


class IdentifyResponse(BaseModel):
    band_match_tolerance: float
    observed_band_count: int
    samples: dict[SampleName, SampleMatchOut]
    verdict: Literal["reference", "variant", "ambiguous"]
    verdict_reason: str = Field(description="中文判定依据说明")


# ---------------------------------------------------------------- 面板（panel）模型

class PanelEnzymeInput(BaseModel):
    name: str = Field(..., min_length=1, description="酶名称，如 EcoRI")
    reference_sites: list[int] = Field(
        default_factory=list, description="参考型上该酶的候选切点位置（bp，0-based）"
    )
    variant_sites: dict[str, list[int]] = Field(
        default_factory=dict,
        description="按变异名映射的候选切点：变异名 -> 该变异型上的切点位置列表（bp，0-based）",
    )


class PanelRequest(BaseModel):
    plasmid_length: int = Field(..., gt=0, description="环状质粒总长度（bp）")
    min_visible_fragment: int = Field(
        ..., ge=0, description="最小可见片段长度（bp），短于此值的片段不参与条带聚类"
    )
    band_resolution_tolerance: int = Field(
        ..., ge=0, description="条带分辨容差（bp），长度差不超过该值的相邻片段聚为同一条带"
    )
    enzymes: list[PanelEnzymeInput] = Field(..., min_length=2, description="参与枚举的酶，至少 2 种")


class PanelVariantCoverageOut(BaseModel):
    variant: str = Field(description="变异名")
    distinguishable_band_count: int = Field(description="该酶对在此变异型上的可区分条带数")
    differences: list[DifferenceOut] = Field(description="逐条差异条带依据（与推荐接口同结构）")


class PanelSelectedPairOut(BaseModel):
    enzyme_pair: list[str]
    pair_key: str
    covered_variants: list[str] = Field(description="该酶对覆盖的变异名，按字典序")
    distinguishable_band_count: int = Field(description="该酶对在全部变异型上的可区分条带总数")
    coverages: list[PanelVariantCoverageOut] = Field(description="逐变异型的差异条带依据")


class PanelResponse(BaseModel):
    plasmid_length: int
    min_visible_fragment: int
    band_resolution_tolerance: int
    variant_count: int = Field(description="请求中全部变异型的数量")
    total_enzyme_pairs: int
    selected_pairs: list[PanelSelectedPairOut] = Field(
        description="最小面板中的酶对，按 pair_key 字典序；存在未覆盖变异型时为空（不返回部分面板）"
    )
    covered_variants: list[str] = Field(description="面板覆盖的变异名；未给出面板时为空")
    uncovered_variants: list[str] = Field(
        description="无任何可用酶对覆盖的变异名；非空时不返回部分面板"
    )
    total_distinguishable_bands: int = Field(description="面板全部酶对的可区分条带总数")


# ---------------------------------------------------------------- 混合（mixture）模型

class MixtureComponentInput(BaseModel):
    name: str = Field(..., min_length=1, description="候选成分名（如某变异型 / 单菌编号）")
    predicted_bands: list[NonNegativeBand] = Field(
        ..., min_length=1, description="该候选成分单独的预测条带长度列表（bp），至少 1 条"
    )


class MixtureRequest(BaseModel):
    components: list[MixtureComponentInput] = Field(
        ..., min_length=1, description="命名候选成分列表，名称不可重复，至少 1 个候选"
    )
    observed_bands: list[NonNegativeBand] = Field(
        ..., min_length=1, description="一次凝胶观测到的混合样本条带长度列表（bp），至少 1 条"
    )
    band_match_tolerance: NonNegativeBand = Field(
        ...,
        description="条带容差（bp）：既用于候选间共迁移条带的单连接聚带，"
                    "也用于合并预测条带与观测条带的一对一配对",
    )
    max_components: int = Field(
        ..., ge=1, description="成分数上限：枚举成分数不超过该值的全部候选子集，且不得超过候选总数"
    )


class MixtureResponse(BaseModel):
    band_match_tolerance: float
    observed_band_count: int
    max_components: int
    selected_components: list[str] = Field(
        description="选中的候选成分名（成分数不超过上限的最优子集），按字典序"
    )
    merged_predicted_bands: list[float] = Field(
        description="选中子集全部预测条带按容差单连接聚带后的代表长度（簇内均值），升序"
    )
    match: SampleMatchOut = Field(
        description="合并预测条带与观测条带的一对一最小代价匹配结果（与鉴别接口同结构）"
    )
    evaluated_subset_count: int = Field(
        description="实际评估的候选子集数量（成分数 1..max_components 的全部非空子集）"
    )
    selection_reason: str = Field(
        description="中文选定依据：未匹配条带数、总误差、成分数及候选名列表如何决定唯一结果"
    )


# ------------------------------------------------- 批量盲样分配（batch-identify）模型

class BatchProfileInput(BaseModel):
    name: str = Field(..., min_length=1, description="命名预测谱型名称（如菌株 / 单菌编号）")
    predicted_bands: list[NonNegativeBand] = Field(
        ..., min_length=1, description="该谱型的预测条带长度列表（bp），至少 1 条"
    )


class BatchLaneInput(BaseModel):
    name: str = Field(..., min_length=1, description="命名观测泳道名称（如凝胶道编号）")
    observed_bands: list[NonNegativeBand] = Field(
        ..., min_length=1, description="该泳道的凝胶观测条带长度列表（bp），至少 1 条"
    )


class BatchIdentifyRequest(BaseModel):
    profiles: list[BatchProfileInput] = Field(
        ..., min_length=1, description="命名预测谱型列表，名称不可重复，至少 1 个"
    )
    lanes: list[BatchLaneInput] = Field(
        ..., min_length=1, description="命名观测泳道列表，名称不可重复，至少 1 个"
    )
    band_match_tolerance: NonNegativeBand = Field(
        ..., description="条带匹配容差（bp）：长度差不超过该值的预测/观测条带才允许配对"
    )
    max_unmatched_bands: int = Field(
        ..., ge=0,
        description="单条分配允许的最大未匹配条带数：未匹配数超过该值的谱型-泳道边不可用",
    )


class BatchAssignmentOut(BaseModel):
    lane: str = Field(description="观测泳道名；assignments 按泳道名字典序排列")
    profile: str = Field(description="全局最优分配中分配给该泳道的预测谱型名")
    unmatched_count: int = Field(
        description="该谱型-泳道边的未匹配条带总数（缺失 + 额外），不超过请求上限"
    )
    match: SampleMatchOut = Field(
        description="该谱型预测条带与该泳道观测条带的一对一最小代价匹配结果"
        "（与鉴别接口同结构，含逐条配对、缺失/额外条带依据）"
    )
    description: str = Field(description="中文分配依据说明")


class BatchIdentifyResponse(BaseModel):
    band_match_tolerance: float
    max_unmatched_bands: int
    profile_count: int
    lane_count: int
    evaluated_edge_count: int = Field(
        description="评估的谱型-泳道候选边总数，等于谱型数 × 泳道数"
    )
    eligible_edge_count: int = Field(
        description="未匹配条带数不超过上限、可参与全局一对一分配的候选边数"
    )
    assignments: list[BatchAssignmentOut] = Field(
        description="逐项匹配依据，按泳道名字典序排列；每个谱型与每个泳道至多出现一次"
    )
    unassigned_lanes: list[str] = Field(
        description="未分配到任何谱型的泳道名，按字典序"
    )
    unused_profiles: list[str] = Field(
        description="未分配给任何泳道的谱型名，按字典序"
    )
    assigned_lane_count: int = Field(description="已分配泳道数（全局最大可分配数）")
    total_unmatched_count: int = Field(description="全部分配边的未匹配条带总数（同级最优中最小）")
    total_error: float = Field(description="全部分配边的总误差之和（bp，同级最优中最小）")
    assignment_reason: str = Field(
        description="中文分配依据：可用边阈值、三级优化目标与字典序破同分如何决定唯一结果"
    )


# ------------------------------------------------- 自适应决策树（decision-tree）模型

class DecisionEnzymeInput(BaseModel):
    name: str = Field(..., min_length=1, description="酶名称，如 EcoRI")
    genotype_sites: dict[str, list[int]] = Field(
        ...,
        description="按基因型名映射的候选切点：基因型名 -> 该基因型上的切点位置列表（bp，0-based）",
    )


class DecisionTreeRequest(BaseModel):
    plasmid_length: int = Field(..., gt=0, description="环状质粒总长度（bp）")
    min_visible_fragment: int = Field(
        ..., ge=0, description="最小可见片段长度（bp），短于此值的片段不参与条带聚类"
    )
    band_resolution_tolerance: int = Field(
        ..., ge=0, description="条带分辨容差（bp）：长度差不超过该值的相邻片段聚为同一条带"
    )
    enzymes: list[DecisionEnzymeInput] = Field(
        ..., min_length=2, description="参与枚举的酶，至少 2 种，名称不可重复"
    )


class DecisionPairOut(BaseModel):
    enzyme_pair: list[str]
    pair_key: str


class DecisionBranchOut(BaseModel):
    signature: list[float] = Field(
        description="该分支对应的规范化条带签名（各条带代表长度，bp，升序）"
    )
    genotypes: list[str] = Field(
        description="落入此签名分支的基因型名（按字典序）；叶节点时恰好 1 个或为不可区分组"
    )
    child: "DecisionTreeNodeOut" = Field(description="该分支的子节点（内部节点或叶节点）")


class DecisionTreeNodeOut(BaseModel):
    node_id: int = Field(description="节点编号，按先根遍历从 1 开始")
    depth: int = Field(description="节点深度：根节点为 0，每做一次实验 +1")
    node_type: Literal["internal", "leaf"]
    candidates: list[str] = Field(
        description="到达该节点时尚待区分的基因型名（按字典序）"
    )
    pair_key: str | None = Field(
        default=None, description="内部节点：本节点选用的酶对（实验）；叶节点为 null"
    )
    enzyme_pair: list[str] | None = Field(
        default=None, description="内部节点：本节点酶对的两个酶名（字典序）；叶节点为 null"
    )
    branches: list[DecisionBranchOut] | None = Field(
        default=None,
        description="内部节点：按观测签名分组的分支，按签名向量字典序排列；叶节点为 null",
    )
    indistinguishable: bool = Field(
        default=False, description="叶节点：候选基因型是否为永远无法区分的不可区分组（多于 1 个）"
    )


class DecisionTreeResponse(BaseModel):
    plasmid_length: int
    min_visible_fragment: int
    band_resolution_tolerance: int
    genotype_count: int = Field(description="请求中的基因型数量")
    genotypes: list[str] = Field(description="全部基因型名（按字典序）")
    total_enzyme_pairs: int = Field(description="不同酶对总数，等于 C(酶数, 2)")
    discriminating_pair_count: int = Field(
        description="至少能拆分某两个基因型、实际参与决策树搜索的酶对数"
    )
    non_discriminating_pairs: list[DecisionPairOut] = Field(
        description="对全部基因型产生相同签名、无法拆分任何集合的酶对（按 pair_key 字典序）"
    )
    tree: DecisionTreeNodeOut = Field(description="自适应决策树根节点")
    worst_case_depth: int = Field(
        description="最坏实验次数：根到叶的最大深度（每一层内部节点为一次双酶切实验）"
    )
    internal_node_count: int = Field(description="决策树内部节点（实验）总数")
    leaf_count: int = Field(description="决策树叶节点总数")
    indistinguishable_group_count: int = Field(description="不可区分组（非单叶）数量")
    indistinguishable_groups: list[list[str]] = Field(
        description="永远无法区分的基因型组：所有酶对上签名都相同的基因型集合"
        "（组内与各组均按字典序排列）；全部基因型可区分时为空"
    )
    tree_reason: str = Field(
        description="中文依据：规范化签名、不可区分合并、极小化极大目标与破同分规则"
    )


DecisionBranchOut.model_rebuild()


# ------------------------------------------------- 酶切部分失败反推（partial-failure）模型

class TheoreticalFragmentInput(BaseModel):
    length: int = Field(
        ..., gt=0, description="按圆环顺序排列的理论片段长度（bp），必须为正数"
    )
    following_boundary: str = Field(
        ..., min_length=1,
        description="该片段后边界的名称（下一个酶切边界）；所有片段的后边界名须唯一",
    )


class PartialFailureRequest(BaseModel):
    plasmid_length: int = Field(..., gt=0, description="环状质粒总长度（bp）")
    fragments: list[TheoreticalFragmentInput] = Field(
        ..., min_length=1,
        description="按圆环顺序排列的理论片段：每段给出长度及其后边界名称；"
                    "全部片段长度之和必须等于质粒长度，后边界名不可重复",
    )
    observed_bands: list[NonNegativeBand] = Field(
        ..., min_length=1, description="一次凝胶观测到的条带长度列表（bp），至少 1 条"
    )
    band_match_tolerance: NonNegativeBand = Field(
        ..., description="条带匹配容差（bp）：长度差不超过该值的预测/观测条带才允许配对"
    )
    max_failed_boundaries: int = Field(
        ..., ge=0,
        description="最多枚举的失败（未切开）边界数：枚举失败边界数 0..该值的全部"
                    "边界子集，且不得超过理论片段（边界）总数",
    )


class PartialFailureMergedBandOut(BaseModel):
    length: float = Field(description="失败边界合并后该预测条带的长度（bp，等于合并片段长度之和）")
    fragment_indexes: list[int] = Field(
        description="合并进该条带的理论片段序号（按请求中的圆环顺序，从 1 开始）"
    )
    fragment_lengths: list[int] = Field(description="各组成理论片段的长度（bp），与片段序号一一对应")
    failed_boundaries_inside: list[str] = Field(
        description="该条带内部被合并跨越的失败边界名（条带末尾保留切开的边界不在内），按圆环顺序"
    )
    crosses_origin: bool = Field(description="该合并条带是否跨过坐标原点（含末段理论片段）")


class PartialFailureResponse(BaseModel):
    plasmid_length: int
    band_match_tolerance: float
    observed_band_count: int
    fragment_count: int = Field(description="请求中的理论片段（边界）数量")
    max_failed_boundaries: int
    evaluated_subset_count: int = Field(
        description="实际评估的失败边界子集数量，等于 Σ C(边界数, k)（k = 0..实际上限）"
    )
    failed_boundary_count: int = Field(description="选定诊断的失败（未切开）边界数量")
    failed_boundaries: list[str] = Field(
        description="选定诊断的失败边界名，按名称字典序排列；完全酶切时为空"
    )
    merged_predicted_bands: list[float] = Field(
        description="失败边界合并相邻片段后的预测条带长度（bp），按长度升序"
    )
    merged_band_details: list[PartialFailureMergedBandOut] = Field(
        description="合并条带的组成依据（按圆环顺序，首条含第 1 个理论片段）"
    )
    match: SampleMatchOut = Field(
        description="合并预测条带与观测条带的一对一最小代价匹配结果（与鉴别接口同结构）"
    )
    selection_reason: str = Field(
        description="中文选定依据：未匹配条带数、总误差、失败边界数及边界名序列如何决定唯一结果"
    )


# ------------------------------------------ 凝胶重复一致性（replicate-consistency）模型

class ReplicateLaneInput(BaseModel):
    name: str = Field(..., min_length=1, description="命名观测泳道名称（如凝胶道编号）")
    observed_bands: list[NonNegativeBand] = Field(
        ..., min_length=1, description="该泳道的凝胶观测条带长度列表（bp），至少 1 条"
    )


class ReplicateConsistencyRequest(BaseModel):
    lanes: list[ReplicateLaneInput] = Field(
        ..., min_length=1,
        description="同一样本的命名观测泳道列表，名称不可重复，至少 1 条",
    )
    band_match_tolerance: NonNegativeBand = Field(
        ..., description="条带匹配容差（bp）：全局聚类时长度差不超过该值的相邻条带归入同一簇"
    )
    min_occurrences: int = Field(
        ..., ge=1,
        description="最少出现次数：簇内不同泳道数达到该值才视为稳定簇（参与缺失判定），"
                    "且不得超过泳道总数",
    )


class ReplicateClusterMemberOut(BaseModel):
    lane: str = Field(description="条带所在泳道名")
    length: float = Field(description="该观测条带长度（bp）")


class ReplicateClusterOut(BaseModel):
    cluster_id: int = Field(description="簇编号，按中位长度升序，从 1 开始")
    median_length: float = Field(description="簇内全部条带长度的中位数（bp，保留两位小数）")
    min_length: float
    max_length: float
    band_count: int = Field(description="簇内条带总数（同一泳道可贡献多条）")
    lane_count: int = Field(description="簇内不同泳道数，即该条带的出现次数")
    is_low_frequency: bool = Field(description="出现次数是否低于最少出现次数（低频簇）")
    lanes: list[str] = Field(description="簇内出现的泳道名，按字典序")
    members: list[ReplicateClusterMemberOut] = Field(
        description="簇内全部条带，按（长度、泳道序号）升序"
    )


class ReplicateLaneReportOut(BaseModel):
    lane: str = Field(description="泳道名；lanes 按泳道名字典序排列")
    band_count: int = Field(description="该泳道观测条带总数")
    matched_band_count: int = Field(description="该泳道作为簇代表参与共识的条带数（每簇至多 1 条）")
    missing_count: int = Field(description="缺失条带数：该泳道未参与的稳定簇数")
    missing_bands: list[float] = Field(description="缺失条带：未覆盖稳定簇的中位长度（bp），升序")
    extra_count: int = Field(description="额外条带数：同簇多条带中除代表条带外的条带数")
    extra_bands: list[float] = Field(description="额外条带：该泳道在同一簇内的非代表条带长度（bp），升序")
    low_frequency_count: int = Field(description="低频条带数：该泳道代表条带落入低频簇的数量")
    low_frequency_bands: list[float] = Field(
        description="低频条带：该泳道落入低频簇的代表条带长度（bp），升序"
    )
    anomaly_count: int = Field(description="异常数：缺失 + 额外 + 低频")
    consensus_deviation: float = Field(
        description="共识偏差：该泳道各簇代表条带与簇中位长度差绝对值之和（bp）"
    )
    stability_score: float = Field(
        description="稳定性评分：100 / (1 + 异常数 + 共识偏差 / (容差 + 1))，满分 100"
    )
    description: str = Field(description="中文一致性评估依据说明")


class ReplicateConsistencyResponse(BaseModel):
    band_match_tolerance: float
    min_occurrences: int
    lane_count: int
    total_band_count: int = Field(description="全部泳道的观测条带总数")
    cluster_count: int = Field(description="全局聚类形成的簇总数")
    stable_cluster_count: int = Field(description="稳定簇数（出现次数达到最少出现次数）")
    low_frequency_cluster_count: int = Field(description="低频簇数（出现次数低于最少出现次数）")
    consensus_bands: list[float] = Field(description="共识谱：每个簇的中位长度（bp），升序")
    clusters: list[ReplicateClusterOut] = Field(
        description="全部簇的组成依据，按中位长度升序"
    )
    lanes: list[ReplicateLaneReportOut] = Field(
        description="逐泳道一致性评估，按泳道名字典序"
    )
    most_stable_lanes: list[str] = Field(
        description="稳定性评分最高的泳道名（并列时全部列出），按字典序"
    )
    stability_reason: str = Field(
        description="中文评估依据：全局聚类、共识谱、异常判定与稳定性评分公式"
    )


# ------------------------------------------------- 迁移距离分子量校准（size-calibration）模型

# 迁移距离：非负有限浮点数（0 为点样孔位置，JSON 整数自动转换）
NonNegativeDistance = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SizeCalibrationRequest(BaseModel):
    ladder_sizes: list[PositiveBand] = Field(
        ..., min_length=2,
        description="标准 ladder 片段尺寸（bp），按尺寸严格降序（即凝胶上迁移距离递增顺序）"
                    "提交，每条 > 0、不可重复，至少 2 条",
    )
    ladder_distances: list[NonNegativeDistance] = Field(
        ..., min_length=2,
        description="ladder 泳道各条带的实测迁移距离（与点样孔的距离，单位任意但全请求一致）；"
                    "允许相对标准 ladder 存在缺带/杂带，距离不可重复，至少 2 条",
    )
    sample_distances: list[NonNegativeDistance] = Field(
        default_factory=list,
        description="待测条带的实测迁移距离列表（与 ladder 同一距离尺度），距离不可重复，可为空列表",
    )
    max_missing_or_spurious_bands: int = Field(
        ..., ge=0,
        description="允许的缺带（未在实测中出现的标准带）与杂带（实测中不属于 ladder 的带）总数上限",
    )
    residual_threshold: NonNegativeBand = Field(
        ...,
        description="入选锚点在迁移距离-log10(bp) 空间线性回归的均方根残差（RMSE，"
                    "log10(bp) 单位）上限；最优锚点对齐的残差超过该值时整体拒绝（422）",
    )


class SizeAnchorOut(BaseModel):
    anchor_index: int = Field(description="锚点编号，按迁移距离升序（log10(bp) 降序），从 1 开始")
    ladder_size: float = Field(description="该锚点对应的标准 ladder 尺寸（bp）")
    ladder_size_index: int = Field(description="该标准带在请求 ladder_sizes 中的序号（降序尺寸序号，从 1 开始）")
    ladder_distance: float = Field(description="该锚点对应的 ladder 实测迁移距离")
    ladder_distance_index: int = Field(description="该实测带在请求 ladder_distances 中的提交序号（从 1 开始）")
    fitted_log10_size: float = Field(description="线性回归拟合的 log10(bp) 值（保留 6 位小数）")
    residual_log10: float = Field(
        description="实际 log10(bp) 减拟合 log10(bp) 的残差（log10(bp) 单位，保留 6 位小数）"
    )
    description: str = Field(description="中文锚点对应依据说明")


class MissingStandardBandOut(BaseModel):
    ladder_size_index: int = Field(description="缺失标准带在请求 ladder_sizes 中的序号（从 1 开始）")
    ladder_size: float = Field(description="缺失标准带的尺寸（bp）")
    description: str = Field(description="中文缺失依据说明")


class SpuriousBandOut(BaseModel):
    ladder_distance_index: int = Field(description="杂带在请求 ladder_distances 中的提交序号（从 1 开始）")
    distance: float = Field(description="杂带的实测迁移距离")
    description: str = Field(description="中文杂带依据说明")


class SizeRegressionOut(BaseModel):
    slope: float = Field(description="线性回归斜率（log10(bp) / 距离单位，距离越大 bp 越小，故为负）")
    intercept: float = Field(description="线性回归截距（log10(bp) 单位）")
    rmse: float = Field(description="锚点 log10(bp) 残差的均方根（RMSE，log10(bp) 单位，保留 6 位小数）")
    r_squared: float = Field(description="线性回归决定系数 R²（保留 6 位小数）")
    residual_threshold: float = Field(description="请求给定的残差阈值（log10(bp) 单位）")
    within_threshold: bool = Field(description="RMSE 是否不超过残差阈值")


class CalibratedSampleBandOut(BaseModel):
    sample_band_index: int = Field(description="待测带在请求 sample_distances 中的提交序号（从 1 开始）")
    distance: float = Field(description="待测带实测迁移距离")
    status: Literal["calibrated", "out_of_range_above", "out_of_range_below"] = Field(
        description="calibrated：落在锚点距离范围内，已分段插值估算；"
                    "out_of_range_above：距离小于最小锚点距离（片段比最大锚点还大，禁止外推）；"
                    "out_of_range_below：距离大于最大锚点距离（片段比最小锚点还小，禁止外推）"
    )
    estimated_size: float | None = Field(
        default=None, description="估算片段尺寸（bp，保留 2 位小数）；越界时为 null"
    )
    segment_anchor_indexes: list[int] | None = Field(
        default=None,
        description="插值所用相邻两个锚点的编号（按距离升序的 anchor_index）；恰好落在锚点上为单元素列表；越界时为 null",
    )
    description: str = Field(description="中文估算 / 越界依据说明")


class SizeCalibrationResponse(BaseModel):
    ladder_band_count: int = Field(description="请求中的标准 ladder 尺寸条数")
    measured_band_count: int = Field(description="请求中的 ladder 实测距离条数")
    sample_band_count: int = Field(description="请求中的待测条带距离条数")
    max_missing_or_spurious_bands: int
    anchor_count: int = Field(description="入选锚点对数（至少 2 个才可校准）")
    missing_standard_count: int = Field(description="缺失标准带数：未入选锚点的标准 ladder 条数")
    spurious_band_count: int = Field(description="杂带数：未入选锚点的 ladder 实测条数")
    skip_total: int = Field(description="跳过总数 = 缺失标准带数 + 杂带数（一级优化目标，越小越优）")
    anchors: list[SizeAnchorOut] = Field(
        description="一对一锚点对应，按迁移距离升序（log10(bp) 降序）排列"
    )
    missing_standard_bands: list[MissingStandardBandOut] = Field(
        description="缺失标准带（未在实测中找到对应的标准 ladder 条带），按 ladder_sizes 序号升序"
    )
    spurious_bands: list[SpuriousBandOut] = Field(
        description="杂带（实测中不属于标准 ladder 的条带），按迁移距离升序"
    )
    regression: SizeRegressionOut = Field(description="入选锚点的线性回归结果与残差阈值校验")
    calibration_range_min_distance: float = Field(description="可校准距离范围下限（最小锚点距离）")
    calibration_range_max_distance: float = Field(description="可校准距离范围上限（最大锚点距离）")
    calibration_range_min_size: float = Field(
        description="可校准 bp 范围下限（最大锚点距离对应的最小锚点 bp）"
    )
    calibration_range_max_size: float = Field(
        description="可校准 bp 范围上限（最小锚点距离对应的最大锚点 bp）"
    )
    sample_bands: list[CalibratedSampleBandOut] = Field(
        description="逐条待测带的估算结果，按请求提交顺序排列"
    )
    calibration_reason: str = Field(
        description="中文依据：锚点枚举约束、跳过总数/残差/索引序列三级优选、禁外推分段插值规则"
    )
