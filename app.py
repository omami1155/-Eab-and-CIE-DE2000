import itertools
import math
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# 色差計算
# =========================
def delta_e_ab(L1, a1, b1, L2, a2, b2):
    """従来の ΔE*ab (CIE76)"""
    return math.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)



def delta_e_00(L1, a1, b1, L2, a2, b2, kL=1, kC=1, kH=1):
    """CIEDE2000 (ΔE00)"""
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - math.sqrt((C_bar**7) / ((C_bar**7) + (25.0**7))))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    def hp(b, ap):
        if ap == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = hp(b1, a1p)
    h2p = hp(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180.0:
            dhp -= 360.0
        elif dhp < -180.0:
            dhp += 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2.0)

    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        h_bar_p = h1p + h2p
    else:
        if abs(h1p - h2p) <= 180.0:
            h_bar_p = (h1p + h2p) / 2.0
        elif (h1p + h2p) < 360.0:
            h_bar_p = (h1p + h2p + 360.0) / 2.0
        else:
            h_bar_p = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    dtheta = 30.0 * math.exp(-((h_bar_p - 275.0) / 25.0) ** 2)
    Rc = 2.0 * math.sqrt((C_bar_p**7) / ((C_bar_p**7) + (25.0**7)))

    Sl = 1.0 + (0.015 * ((L_bar_p - 50.0) ** 2)) / math.sqrt(20.0 + ((L_bar_p - 50.0) ** 2))
    Sc = 1.0 + 0.045 * C_bar_p
    Sh = 1.0 + 0.015 * C_bar_p * T
    Rt = -math.sin(math.radians(2.0 * dtheta)) * Rc

    return math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


# =========================
# 共通ユーティリティ
# =========================
def load_csv(uploaded_file):
    encodings = ["utf-8-sig", "utf-8", "cp932"]
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSVを読み込めませんでした。UTF-8 / UTF-8-SIG / CP932 を確認してください。")



def format_num(value, digits=4):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"



def coerce_numeric_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out



def rounded_display(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        out[col] = out[col].round(digits)
    return out



def make_distance_matrix(lab_df: pd.DataFrame) -> pd.DataFrame:
    n = len(lab_df)
    ids = lab_df["SampleID"].tolist()
    arr = lab_df[["L_mean", "a_mean", "b_mean"]].to_numpy(dtype=float)

    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            de00 = delta_e_00(
                arr[i, 0], arr[i, 1], arr[i, 2],
                arr[j, 0], arr[j, 1], arr[j, 2]
            )
            dist[i, j] = de00
            dist[j, i] = de00

    return pd.DataFrame(dist, index=ids, columns=ids)



def pairwise_deltae_table(sample_means: pd.DataFrame, g1: str, g2: str) -> pd.DataFrame:
    left = sample_means[sample_means["Group"] == g1].copy()
    right = sample_means[sample_means["Group"] == g2].copy()

    rows = []
    for _, r1 in left.iterrows():
        for _, r2 in right.iterrows():
            rows.append({
                "群1": g1,
                "サンプル1": r1["SampleID"],
                "群2": g2,
                "サンプル2": r2["SampleID"],
                "ΔE00": delta_e_00(
                    r1["L_mean"], r1["a_mean"], r1["b_mean"],
                    r2["L_mean"], r2["a_mean"], r2["b_mean"],
                ),
                "ΔE*ab": delta_e_ab(
                    r1["L_mean"], r1["a_mean"], r1["b_mean"],
                    r2["L_mean"], r2["a_mean"], r2["b_mean"],
                ),
            })
    return pd.DataFrame(rows)



def within_group_deltae_table(sample_means: pd.DataFrame, group_name: str) -> pd.DataFrame:
    sub = sample_means[sample_means["Group"] == group_name].copy()
    rows = []
    for i, j in itertools.combinations(sub.index, 2):
        r1 = sub.loc[i]
        r2 = sub.loc[j]
        rows.append({
            "群": group_name,
            "サンプル1": r1["SampleID"],
            "サンプル2": r2["SampleID"],
            "ΔE00": delta_e_00(
                r1["L_mean"], r1["a_mean"], r1["b_mean"],
                r2["L_mean"], r2["a_mean"], r2["b_mean"],
            ),
            "ΔE*ab": delta_e_ab(
                r1["L_mean"], r1["a_mean"], r1["b_mean"],
                r2["L_mean"], r2["a_mean"], r2["b_mean"],
            ),
        })
    return pd.DataFrame(rows)


# =========================
# PERMANOVA
# =========================
def permanova_ss(distance_matrix: np.ndarray, groups: Sequence[str]) -> Tuple[float, float, float]:
    """1要因PERMANOVAのSS群間, SS群内, SS全体を返す"""
    n = len(groups)
    unique_groups = list(pd.unique(pd.Series(groups)))

    sst = distance_matrix[np.triu_indices(n, 1)]
    ss_total = np.sum(sst ** 2) / n

    ss_within = 0.0
    for g in unique_groups:
        idx = [i for i, x in enumerate(groups) if x == g]
        nk = len(idx)
        if nk <= 1:
            continue
        sub = distance_matrix[np.ix_(idx, idx)]
        upper = sub[np.triu_indices(nk, 1)]
        ss_within += np.sum(upper ** 2) / nk

    ss_between = ss_total - ss_within
    return ss_between, ss_within, ss_total



def permanova_pseudo_f(distance_matrix: np.ndarray, groups: Sequence[str]) -> Tuple[float, float, float]:
    unique_groups = list(pd.unique(pd.Series(groups)))
    n = len(groups)
    g = len(unique_groups)

    ss_between, ss_within, _ = permanova_ss(distance_matrix, groups)

    df_between = g - 1
    df_within = n - g

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    pseudo_f = ms_between / ms_within if ms_within > 0 else np.nan
    r2 = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else np.nan
    return pseudo_f, r2, ms_within



def exact_or_monte_carlo_permanova(
    distance_matrix: np.ndarray,
    groups: Sequence[str],
    max_exact_partitions: int = 50000,
    monte_carlo_permutations: int = 9999,
    random_seed: int = 42,
) -> dict:
    groups = list(groups)
    unique_groups = list(pd.unique(pd.Series(groups)))
    if len(unique_groups) != 2:
        raise ValueError("このアプリの主解析は2群比較のみ対応しています。")

    n = len(groups)
    n_a = sum(1 for x in groups if x == unique_groups[0])
    observed_f, r2, ms_within = permanova_pseudo_f(distance_matrix, groups)

    total_partitions = math.comb(n, n_a)

    perm_f_values = []
    if total_partitions <= max_exact_partitions:
        all_indices = list(range(n))
        for comb in itertools.combinations(all_indices, n_a):
            perm_groups = [unique_groups[1]] * n
            for idx in comb:
                perm_groups[idx] = unique_groups[0]
            f_val, _, _ = permanova_pseudo_f(distance_matrix, perm_groups)
            perm_f_values.append(f_val)
        method = f"厳密置換 ({total_partitions}通り)"
    else:
        rng = np.random.default_rng(random_seed)
        arr_groups = np.array(groups)
        for _ in range(monte_carlo_permutations):
            permuted = rng.permutation(arr_groups)
            f_val, _, _ = permanova_pseudo_f(distance_matrix, permuted.tolist())
            perm_f_values.append(f_val)
        method = f"モンテカルロ置換 ({monte_carlo_permutations}回)"

    perm_f_values = np.array(perm_f_values, dtype=float)
    p_value = (1 + np.sum(perm_f_values >= observed_f)) / (1 + len(perm_f_values))

    return {
        "pseudo_F": observed_f,
        "R2": r2,
        "p_value": p_value,
        "perm_method": method,
        "n_permutations": len(perm_f_values),
        "ms_within": ms_within,
    }



def permutation_test_two_groups(values: Sequence[float], groups: Sequence[str]) -> dict:
    """2群の差の置換検定（平均差）"""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique_groups = pd.unique(pd.Series(groups))
    if len(unique_groups) != 2:
        raise ValueError("2群比較のみ対応しています。")

    g1, g2 = unique_groups[0], unique_groups[1]
    idx1 = np.where(groups == g1)[0]
    idx2 = np.where(groups == g2)[0]

    obs_diff = np.mean(values[idx1]) - np.mean(values[idx2])

    n = len(values)
    n1 = len(idx1)
    diffs = []
    for comb in itertools.combinations(range(n), n1):
        mask = np.zeros(n, dtype=bool)
        mask[list(comb)] = True
        diff = np.mean(values[mask]) - np.mean(values[~mask])
        diffs.append(diff)

    diffs = np.asarray(diffs, dtype=float)
    p_value = (1 + np.sum(np.abs(diffs) >= abs(obs_diff))) / (1 + len(diffs))

    return {
        "group1": g1,
        "group2": g2,
        "mean_group1": float(np.mean(values[idx1])),
        "mean_group2": float(np.mean(values[idx2])),
        "difference": float(obs_diff),
        "p_value": float(p_value),
        "n_permutations": int(len(diffs)),
    }


# =========================
# データ処理
# =========================
def summarize_samples(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["Group", "SampleID"], dropna=False)
        .agg(
            n_repeats=("L*", "size"),
            L_mean=("L*", "mean"),
            a_mean=("a*", "mean"),
            b_mean=("b*", "mean"),
            L_sd=("L*", "std"),
            a_sd=("a*", "std"),
            b_sd=("b*", "std"),
        )
        .reset_index()
    )
    return grouped



def summarize_groups(sample_means: pd.DataFrame) -> pd.DataFrame:
    return (
        sample_means.groupby("Group", dropna=False)
        .agg(
            n_samples=("SampleID", "size"),
            L_group_mean=("L_mean", "mean"),
            a_group_mean=("a_mean", "mean"),
            b_group_mean=("b_mean", "mean"),
            L_group_sd=("L_mean", "std"),
            a_group_sd=("a_mean", "std"),
            b_group_sd=("b_mean", "std"),
        )
        .reset_index()
    )



def validate_input(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    required_cols = ["Group", "SampleID", "L*", "a*", "b*"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"必要な列が不足しています: {', '.join(missing)}")
        return False, errors

    work = coerce_numeric_columns(df, ["L*", "a*", "b*"])
    if work[["L*", "a*", "b*"]].isna().any().any():
        bad_rows = work[work[["L*", "a*", "b*"]].isna().any(axis=1)].index.tolist()
        errors.append(f"L*, a*, b* に数値でない値があります。該当行: {', '.join(str(i + 1) for i in bad_rows[:10])}")

    n_groups = work["Group"].astype(str).nunique()
    if n_groups != 2:
        errors.append(f"Group はちょうど2群必要です。現在: {n_groups} 群")

    return len(errors) == 0, errors



def sample_means_display(sample_means: pd.DataFrame) -> pd.DataFrame:
    disp = sample_means.rename(columns={
        "Group": "群",
        "SampleID": "サンプルID",
        "n_repeats": "測定回数",
        "L_mean": "L*平均",
        "a_mean": "a*平均",
        "b_mean": "b*平均",
        "L_sd": "L*SD",
        "a_sd": "a*SD",
        "b_sd": "b*SD",
        "repeat_warning": "注意",
    })
    return rounded_display(disp)



def group_summary_display(group_summary: pd.DataFrame) -> pd.DataFrame:
    disp = group_summary.rename(columns={
        "Group": "群",
        "n_samples": "サンプル数",
        "L_group_mean": "L*群平均",
        "a_group_mean": "a*群平均",
        "b_group_mean": "b*群平均",
        "L_group_sd": "L*群内SD",
        "a_group_sd": "a*群内SD",
        "b_group_sd": "b*群内SD",
    })
    return rounded_display(disp)



def distance_matrix_display(dist_df: pd.DataFrame) -> pd.DataFrame:
    out = rounded_display(dist_df)
    out.index.name = "サンプルID"
    return out



def style_download_tables(output_dict: dict) -> dict:
    styled = {}
    for key, value in output_dict.items():
        if isinstance(value, pd.DataFrame):
            styled[key] = rounded_display(value)
        else:
            styled[key] = value
    return styled


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="独立二群 色差解析", page_icon="🎨", layout="wide")

st.title("独立二群の色差解析")
st.write(
    "反復測定した L*, a*, b* のCSVを読み込み、"
    "サンプル平均 → 主解析（PERMANOVA）→ 補助解析（群代表色のΔE00、群内/群間の総当たりΔE）まで一括で計算します。"
)

st.subheader("入力CSVの列")
st.code("Group, SampleID, L*, a*, b*")
st.caption("1行 = 1回の測定です。Replicate列は不要です。A1を5回測ったなら、SampleID=A1の行を5行入れてください。")

with st.expander("CSVの例"):
    sample_df = pd.DataFrame([
        {"Group": "A", "SampleID": "A1", "L*": 70.10, "a*": 1.20, "b*": 12.00},
        {"Group": "A", "SampleID": "A1", "L*": 70.30, "a*": 1.00, "b*": 11.90},
        {"Group": "A", "SampleID": "A1", "L*": 70.00, "a*": 1.10, "b*": 12.10},
        {"Group": "A", "SampleID": "A1", "L*": 70.20, "a*": 1.10, "b*": 12.00},
        {"Group": "A", "SampleID": "A1", "L*": 70.10, "a*": 1.20, "b*": 11.80},
        {"Group": "A", "SampleID": "A2", "L*": 69.70, "a*": 1.30, "b*": 11.60},
        {"Group": "A", "SampleID": "A2", "L*": 69.80, "a*": 1.20, "b*": 11.50},
        {"Group": "B", "SampleID": "B1", "L*": 72.40, "a*": 0.60, "b*": 14.20},
        {"Group": "B", "SampleID": "B1", "L*": 72.30, "a*": 0.70, "b*": 14.10},
        {"Group": "B", "SampleID": "B2", "L*": 72.00, "a*": 0.50, "b*": 13.90},
        {"Group": "B", "SampleID": "B2", "L*": 72.20, "a*": 0.60, "b*": 14.00},
    ])
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    st.download_button(
        "サンプルCSVをダウンロード",
        data=sample_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="independent_groups_lab_sample.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = load_csv(uploaded_file)
        st.success("CSVを読み込みました。必要なら表内で直接修正できます。")

        edited_df = st.data_editor(
            raw_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )

        is_valid, errors = validate_input(edited_df)
        if not is_valid:
            for msg in errors:
                st.error(msg)
        else:
            if st.button("解析する", type="primary"):
                work_df = edited_df.copy()
                work_df["Group"] = work_df["Group"].astype(str)
                work_df["SampleID"] = work_df["SampleID"].astype(str)
                work_df = coerce_numeric_columns(work_df, ["L*", "a*", "b*"])
                work_df = work_df.dropna(subset=["L*", "a*", "b*"]).reset_index(drop=True)

                # サンプル平均
                sample_means = summarize_samples(work_df)
                sample_means["repeat_warning"] = np.where(
                    sample_means["n_repeats"] != 5,
                    "※5回ではありません",
                    ""
                )

                # 群代表色
                group_summary = summarize_groups(sample_means)
                if len(group_summary) != 2:
                    st.error("2群比較のみ対応しています。Group を2群にしてください。")
                    st.stop()

                g1 = group_summary.loc[0, "Group"]
                g2 = group_summary.loc[1, "Group"]

                de00_group = delta_e_00(
                    group_summary.loc[0, "L_group_mean"],
                    group_summary.loc[0, "a_group_mean"],
                    group_summary.loc[0, "b_group_mean"],
                    group_summary.loc[1, "L_group_mean"],
                    group_summary.loc[1, "a_group_mean"],
                    group_summary.loc[1, "b_group_mean"],
                )
                de76_group = delta_e_ab(
                    group_summary.loc[0, "L_group_mean"],
                    group_summary.loc[0, "a_group_mean"],
                    group_summary.loc[0, "b_group_mean"],
                    group_summary.loc[1, "L_group_mean"],
                    group_summary.loc[1, "a_group_mean"],
                    group_summary.loc[1, "b_group_mean"],
                )

                # 距離行列とPERMANOVA
                dist_df = make_distance_matrix(sample_means)
                permanova_result = exact_or_monte_carlo_permanova(
                    distance_matrix=dist_df.to_numpy(dtype=float),
                    groups=sample_means["Group"].tolist(),
                )

                # 軸ごとの補助検定
                l_test = permutation_test_two_groups(sample_means["L_mean"], sample_means["Group"])
                a_test = permutation_test_two_groups(sample_means["a_mean"], sample_means["Group"])
                b_test = permutation_test_two_groups(sample_means["b_mean"], sample_means["Group"])

                # 総当たりΔE
                within_g1 = within_group_deltae_table(sample_means, g1)
                within_g2 = within_group_deltae_table(sample_means, g2)
                between = pairwise_deltae_table(sample_means, g1, g2)

                st.divider()
                st.subheader("1) サンプル平均（主解析に使う表）")
                st.dataframe(sample_means_display(sample_means), use_container_width=True, hide_index=True)

                st.subheader("2) 主解析：PERMANOVA（ΔE00距離）")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("擬似F値", format_num(permanova_result["pseudo_F"]))
                c2.metric("p値", format_num(permanova_result["p_value"]))
                c3.metric("R²", format_num(permanova_result["R2"]))
                c4.metric("置換法", permanova_result["perm_method"])

                st.caption(
                    "主解析では、各サンプルの平均L*・a*・b*から 10×10 のΔE00距離行列を作成し、"
                    "群（A/B）でPERMANOVAを行っています。"
                )

                st.subheader("3) 群代表色（補助結果）")
                st.dataframe(group_summary_display(group_summary), use_container_width=True, hide_index=True)

                c5, c6 = st.columns(2)
                c5.metric("群代表色どうしの ΔE00", format_num(de00_group))
                c6.metric("群代表色どうしの ΔE*ab", format_num(de76_group))

                st.subheader("4) L*・a*・b* の個別比較（補助結果）")
                axis_df = pd.DataFrame([
                    {
                        "色座標": "L*",
                        f"{l_test['group1']}群平均": l_test["mean_group1"],
                        f"{l_test['group2']}群平均": l_test["mean_group2"],
                        "平均差（前者 - 後者）": l_test["difference"],
                        "p値": l_test["p_value"],
                        "置換数": l_test["n_permutations"],
                    },
                    {
                        "色座標": "a*",
                        f"{a_test['group1']}群平均": a_test["mean_group1"],
                        f"{a_test['group2']}群平均": a_test["mean_group2"],
                        "平均差（前者 - 後者）": a_test["difference"],
                        "p値": a_test["p_value"],
                        "置換数": a_test["n_permutations"],
                    },
                    {
                        "色座標": "b*",
                        f"{b_test['group1']}群平均": b_test["mean_group1"],
                        f"{b_test['group2']}群平均": b_test["mean_group2"],
                        "平均差（前者 - 後者）": b_test["difference"],
                        "p値": b_test["p_value"],
                        "置換数": b_test["n_permutations"],
                    },
                ])
                st.dataframe(rounded_display(axis_df), use_container_width=True, hide_index=True)

                st.subheader("5) 群内・群間の総当たりΔE（記述用）")
                t1, t2, t3 = st.tabs(
                    [f"{g1}群内", f"{g2}群内", f"{g1}-{g2}群間"]
                )
                with t1:
                    st.dataframe(rounded_display(within_g1), use_container_width=True, hide_index=True)
                with t2:
                    st.dataframe(rounded_display(within_g2), use_container_width=True, hide_index=True)
                with t3:
                    st.dataframe(rounded_display(between), use_container_width=True, hide_index=True)

                summary_rows = []
                for label, sub_df in [
                    (f"{g1}群内", within_g1),
                    (f"{g2}群内", within_g2),
                    (f"{g1}-{g2}群間", between),
                ]:
                    summary_rows.append({
                        "区分": label,
                        "組み合わせ数": len(sub_df),
                        "ΔE00平均": sub_df["ΔE00"].mean() if len(sub_df) else np.nan,
                        "ΔE00中央値": sub_df["ΔE00"].median() if len(sub_df) else np.nan,
                        "ΔE00 SD": sub_df["ΔE00"].std() if len(sub_df) else np.nan,
                        "ΔE00最小": sub_df["ΔE00"].min() if len(sub_df) else np.nan,
                        "ΔE00最大": sub_df["ΔE00"].max() if len(sub_df) else np.nan,
                    })

                pairwise_summary_df = pd.DataFrame(summary_rows)
                st.subheader("6) 総当たりΔEの要約")
                st.dataframe(rounded_display(pairwise_summary_df), use_container_width=True, hide_index=True)
                st.caption(
                    "群内10通り・群間25通りの総当たりΔEは記述用です。"
                    "同じサンプルが繰り返し使われるため、独立nとして推測統計には使いません。"
                )

                st.subheader("7) 距離行列（ΔE00）")
                st.dataframe(distance_matrix_display(dist_df), use_container_width=True)

                permanova_df = pd.DataFrame([
                    {
                        "擬似F値": permanova_result["pseudo_F"],
                        "p値": permanova_result["p_value"],
                        "R²": permanova_result["R2"],
                        "置換法": permanova_result["perm_method"],
                        "置換回数": permanova_result["n_permutations"],
                        "群内平均平方": permanova_result["ms_within"],
                    }
                ])

                # ダウンロード用データ
                output_dict = {
                    "サンプル平均": sample_means_display(sample_means),
                    "群代表色": group_summary_display(group_summary),
                    "色座標別比較": rounded_display(axis_df),
                    f"{g1}群内総当たり": rounded_display(within_g1),
                    f"{g2}群内総当たり": rounded_display(within_g2),
                    f"{g1}_{g2}群間総当たり": rounded_display(between),
                    "距離行列_ΔE00": distance_matrix_display(dist_df).reset_index(),
                    "総当たり要約": rounded_display(pairwise_summary_df),
                    "PERMANOVA結果": rounded_display(permanova_df),
                }

                csv_sample = sample_means_display(sample_means).to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "サンプル平均CSVをダウンロード",
                    data=csv_sample,
                    file_name="sample_means_jp.csv",
                    mime="text/csv",
                )

                excel_path = "independent_groups_color_analysis_jp.xlsx"
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    for sheet_name, out_df in output_dict.items():
                        out_df.to_excel(writer, index=False, sheet_name=sheet_name[:31])

                with open(excel_path, "rb") as f:
                    st.download_button(
                        "解析結果Excelをダウンロード",
                        data=f.read(),
                        file_name=excel_path,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                st.info(
                    "見方の基本: 主解析は PERMANOVA の p値を確認します。"
                    "補助として、群代表色どうしのΔE00、L*・a*・b*の個別比較、"
                    "群内・群間の総当たりΔE分布を見ます。"
                )

    except Exception as e:
        st.error(f"読み込みまたは解析エラー: {e}")
