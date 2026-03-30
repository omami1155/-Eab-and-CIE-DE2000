import math
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


st.set_page_config(page_title="ΔE Calculator", page_icon="🎨", layout="wide")

st.title("色差計算")
st.write("2組の L*, a*, b* から ΔE*ab（CIE76）と ΔE00（CIEDE2000）を計算します。")

with st.form("lab_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("色1")
        L1 = st.number_input("L1*", value=50.0000, format="%.4f")
        a1 = st.number_input("a1*", value=2.6772, format="%.4f")
        b1 = st.number_input("b1*", value=-79.7751, format="%.4f")

    with col2:
        st.subheader("色2")
        L2 = st.number_input("L2*", value=50.0000, format="%.4f")
        a2 = st.number_input("a2*", value=0.0000, format="%.4f")
        b2 = st.number_input("b2*", value=-82.7485, format="%.4f")

    submitted = st.form_submit_button("計算する")

if submitted:
    de76 = delta_e_ab(L1, a1, b1, L2, a2, b2)
    de00 = delta_e_00(L1, a1, b1, L2, a2, b2)

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        st.metric("ΔE*ab (CIE76)", f"{de76:.4f}")
    with r2:
        st.metric("ΔE00 (CIEDE2000)", f"{de00:.4f}")

    with st.expander("入力値の確認"):
        st.write(
            {
                "色1": {"L*": L1, "a*": a1, "b*": b1},
                "色2": {"L*": L2, "a*": a2, "b*": b2},
            }
        )

    st.caption("初期値は CIEDE2000 の代表的な検算例です。")
