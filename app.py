st.markdown("""
<style>

/* --- Fondo general --- */
.stApp {
    background-color: #fffde7;
}

/* --- Encabezado vino tinto fijo --- */
.header-fixed {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 120px;
    background-color: #8b2b2b;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: 800;
    font-family: 'Segoe UI', sans-serif;
    z-index: 1000;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

/* --- Ajuste del contenido principal para no quedar debajo del header --- */
.block-container {
    margin-top: 140px;
    background-color: rgba(0, 0, 0, 0.03);
    padding: 2rem 2rem 3rem 2rem;
    border-radius: 16px;
}

/* --- Títulos --- */
h1, h2, h3, h4, p, span, li, label {
    font-family: 'Segoe UI', sans-serif;
    color: #111111;
}
h1 {
    color: #8b2b2b !important;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background-color: #161921;
    border-right: 3px solid #8b2b2b;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* --- Botones --- */
div.stButton > button:first-child,
section[data-testid="stFileUploader"] button {
    background-color: #8b2b2b !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border: 1.5px solid #5c1a1a !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    text-shadow: 0px 0px 3px rgba(0,0,0,0.4);
    box-shadow: 1px 2px 5px rgba(0,0,0,0.4);
    transition: all 0.25s ease-in-out !important;
}
div.stButton > button:first-child:hover,
section[data-testid="stFileUploader"] button:hover {
    background-color: #6A0000 !important;
    color: #FFD700 !important;
    transform: scale(1.05);
}

/* --- Caja resultado --- */
.result-box {
    background-color: rgba(0, 0, 0, 0.6);
    border: 2px solid #FFD700;
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #ffffff;
}

/* --- Marca de agua --- */
.watermark {
    position: fixed;
    left: 50%;
    bottom: 20px;
    transform: translateX(-50%);
    font-size: 15px;
    font-weight: 600;
    color: rgba(0, 0, 0, 0.8);
    z-index: 9999;
}

</style>
""", unsafe_allow_html=True)



