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
            "status": "OK",
        })
    except Exception:
        return pd.Series({
            "ΔE*ab (CIE76)": None,
            "ΔE00 (CIEDE2000)": None,
            "status": "NG",
        })


def load_csv(uploaded_file):
    # 文字コードの違いに少し強くしておく
    encodings = ["utf-8-sig", "utf-8", "cp932"]
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSVを読み込めませんでした。UTF-8 または CP932 のCSVを確認してください。")


st.set_page_config(page_title="ΔE Calculator CSV", page_icon="🎨", layout="wide")

st.title("色差(CIE76,CIEDE2000)")
st.write("CSVを読み込んで、ΔE*ab（CIE76）と ΔE00（CIEDE2000）を一括計算します。")

st.subheader("必要な列名")
st.code("L1*, a1*, b1*, L2*, a2*, b2*")

with st.expander("CSVの例"):
    sample_df = pd.DataFrame([
        {"ID": "sample1", "L1*": 50.0000, "a1*": 2.6772, "b1*": -79.7751, "L2*": 50.0000, "a2*": 0.0000, "b2*": -82.7485},
        {"ID": "sample2", "L1*": 50.0000, "a1*": 3.1571, "b1*": -77.2803, "L2*": 50.0000, "a2*": 0.0000, "b2*": -82.7485},
    ])
    st.dataframe(sample_df, use_container_width=True)
    sample_csv = sample_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "サンプルCSVをダウンロード",
        data=sample_csv,
        file_name="delta_e_sample.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("CSVファイルを選択してください", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.success("CSVを読み込みました。必要なら表内で直接修正できます。")

        st.subheader("読み込んだデータ")
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )

        required_cols = ["L1*", "a1*", "b1*", "L2*", "a2*", "b2*"]
        missing_cols = [c for c in required_cols if c not in edited_df.columns]

        if missing_cols:
            st.error(f"必要な列が不足しています: {', '.join(missing_cols)}")
        else:
            if st.button("一括計算", type="primary"):
                result_df = edited_df.copy()
                calc_results = result_df.apply(calc_row, axis=1)
                result_df = pd.concat([result_df, calc_results], axis=1)

                st.divider()
                st.subheader("計算結果")
                st.dataframe(result_df, use_container_width=True)

                valid_de00 = result_df["ΔE00 (CIEDE2000)"].dropna()
                valid_de76 = result_df["ΔE*ab (CIE76)"].dropna()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("計算成功件数", int((result_df["status"] == "OK").sum()))
                c2.metric("ΔE00 平均", f"{valid_de00.mean():.4f}" if len(valid_de00) else "-")
                c3.metric("ΔE00 最小", f"{valid_de00.min():.4f}" if len(valid_de00) else "-")
                c4.metric("ΔE00 最大", f"{valid_de00.max():.4f}" if len(valid_de00) else "-")

                csv_result = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="結果CSVをダウンロード",
                    data=csv_result,
                    file_name="delta_e_results.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"読み込みエラー: {e}")
