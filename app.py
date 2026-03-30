import math
import pandas as pd
import streamlit as st


def delta_e_ab(L1, a1, b1, L2, a2, b2):
    """従来の ΔE*ab (CIE76)"""
    return math.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)


def delta_e_00(L1, a1, b1, L2, a2, b2, kL=1, kC=1, kH=1):
    """CIEDE2000 (ΔE00)"""
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - math.sqrt((C_bar ** 7) / ((C_bar ** 7) + (25.0 ** 7))))
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
    Rc = 2.0 * math.sqrt((C_bar_p ** 7) / ((C_bar_p ** 7) + (25.0 ** 7)))

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


def calc_row(row):
    """1行分の計算"""
    try:
        L1 = float(row["L1*"])
        a1 = float(row["a1*"])
        b1 = float(row["b1*"])
        L2 = float(row["L2*"])
        a2 = float(row["a2*"])
        b2 = float(row["b2*"])

        de76 = delta_e_ab(L1, a1, b1, L2, a2, b2)
        de00 = delta_e_00(L1, a1, b1, L2, a2, b2)

        return pd.Series({
            "ΔE*ab (CIE76)": round(de76, 4),
            "ΔE00 (CIEDE2000)": round(de00, 4),
        })
    except Exception:
        return pd.Series({
            "ΔE*ab (CIE76)": None,
            "ΔE00 (CIEDE2000)": None,
        })


st.set_page_config(page_title="ΔE Calculator", page_icon="🎨", layout="wide")

st.title("色差一括計算")
st.write("複数組の L*, a*, b* をまとめて入力して、ΔE*ab（CIE76）と ΔE00（CIEDE2000）を一括計算します。")

default_df = pd.DataFrame([
    {
        "ID": "sample1",
        "L1*": 50.0000,
        "a1*": 2.6772,
        "b1*": -79.7751,
        "L2*": 50.0000,
        "a2*": 0.0000,
        "b2*": -82.7485,
    },
    {
        "ID": "sample2",
        "L1*": 50.0000,
        "a1*": 3.1571,
        "b1*": -77.2803,
        "L2*": 50.0000,
        "a2*": 0.0000,
        "b2*": -82.7485,
    },
])

st.subheader("入力表")
edited_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
)

col1, col2 = st.columns([1, 1])

with col1:
    calc_button = st.button("一括計算", type="primary")

with col2:
    st.caption("行を追加して複数サンプルをまとめて計算できます。")

if calc_button:
    required_cols = ["L1*", "a1*", "b1*", "L2*", "a2*", "b2*"]
    missing_cols = [c for c in required_cols if c not in edited_df.columns]

    if missing_cols:
        st.error(f"必要な列がありません: {', '.join(missing_cols)}")
    else:
        result_df = edited_df.copy()
        calc_results = result_df.apply(calc_row, axis=1)
        result_df = pd.concat([result_df, calc_results], axis=1)

        st.divider()
        st.subheader("計算結果")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="結果をCSVでダウンロード",
            data=csv,
            file_name="delta_e_results.csv",
            mime="text/csv",
        )

        st.subheader("概要")
        valid_de00 = result_df["ΔE00 (CIEDE2000)"].dropna()
        valid_de76 = result_df["ΔE*ab (CIE76)"].dropna()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("計算件数", len(valid_de00))
        with c2:
            st.metric("ΔE00 平均", f"{valid_de00.mean():.4f}" if len(valid_de00) else "-")
        with c3:
            st.metric("ΔE00 最小", f"{valid_de00.min():.4f}" if len(valid_de00) else "-")
        with c4:
            st.metric("ΔE00 最大", f"{valid_de00.max():.4f}" if len(valid_de00) else "-")

        st.caption("空欄や不正値を含む行は結果が空欄になります。")
