from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# MobileSense - Smart Mobile Price Prediction
# Put your trained model in models/best_svc_model.pkl
# Put your dataset in data/train (1).csv (optional; for defaults)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = MODEL_DIR / "best_svc_model.pkl"
DATASET_PATH = DATA_DIR / "train (1).csv"

# Canonical Kaggle Mobile Price Classification input columns.
# If your fitted estimator exposes feature_names_in_, those are used instead.
STANDARD_FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc",
    "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
    "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
    "talk_time", "three_g", "touch_screen", "wifi",
]
TARGET_COLUMN = "price_range"

FEATURE_INFO = {
    "battery_power": ("Battery power (mAh)", 500, 2000, 1000, "Battery capacity in mAh."),
    "blue": ("Bluetooth", 0, 1, 1, "Bluetooth support."),
    "clock_speed": ("Clock speed (GHz)", 0.5, 3.0, 1.5, "Processor clock speed."),
    "dual_sim": ("Dual SIM", 0, 1, 1, "Supports two SIM cards."),
    "fc": ("Front camera (MP)", 0, 20, 5, "Front camera megapixels."),
    "four_g": ("4G support", 0, 1, 1, "Supports 4G."),
    "int_memory": ("Internal memory (GB)", 2, 64, 32, "Internal memory value as encoded in the dataset."),
    "m_dep": ("Mobile depth (cm)", 0.1, 1.0, 0.5, "Phone thickness/depth."),
    "mobile_wt": ("Mobile weight (g)", 80, 200, 150, "Phone weight in grams."),
    "n_cores": ("Processor cores", 1, 8, 4, "Number of processor cores."),
    "pc": ("Primary camera (MP)", 0, 20, 10, "Primary camera megapixels."),
    "px_height": ("Pixel resolution height", 0, 1960, 800, "Pixel height from the dataset."),
    "px_width": ("Pixel resolution width", 500, 2000, 1200, "Pixel width from the dataset."),
    "ram": ("RAM (MB)", 256, 4000, 2000, "RAM amount in MB."),
    "sc_h": ("Screen height (cm)", 5, 19, 12, "Screen height feature from the dataset."),
    "sc_w": ("Screen width (cm)", 0, 18, 7, "Screen width feature from the dataset."),
    "talk_time": ("Talk time (hours)", 2, 20, 10, "Maximum talk time in hours."),
    "three_g": ("3G support", 0, 1, 1, "Supports 3G."),
    "touch_screen": ("Touchscreen", 0, 1, 1, "Has a touchscreen."),
    "wifi": ("Wi-Fi", 0, 1, 1, "Supports Wi-Fi."),
}
MAIN_FEATURES = [
    "ram", "battery_power", "px_height", "px_width",
    "int_memory", "pc", "fc", "n_cores",
]
BINARY_FEATURES = {"blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"}

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="MobileSense | Mobile Price Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7faff; color: #17345c; }
    [data-testid="stHeader"] { background: rgba(247,250,255,.92); }
    [data-testid="stSidebar"] { background: #edf5ff; border-right: 1px solid #d8e7fb; }
    .ms-hero {
        background: linear-gradient(120deg, #eaf3ff 0%, #ffffff 78%);
        border: 1px solid #d8e7fb; border-radius: 22px;
        padding: 30px 32px; margin-bottom: 20px;
    }
    .ms-eyebrow { color: #2563eb; font-weight: 750; letter-spacing: .08em;
        text-transform: uppercase; font-size: .78rem; }
    .ms-title { color: #12305a; font-size: 2.7rem; font-weight: 850;
        line-height: 1.12; margin: 8px 0; }
    .ms-subtitle { color: #526b8b; font-size: 1.05rem; margin-bottom: 0; }
    .ms-card { background: #fff; border: 1px solid #dce8f7; border-radius: 16px;
        padding: 20px; box-shadow: 0 5px 18px rgba(30,64,110,.05); min-height: 120px; }
    .ms-card h3 { color: #12305a; margin: 0 0 8px 0; font-size: 1.05rem; }
    .ms-card p { color: #5b6f89; margin: 0; font-size: .94rem; }
    .ms-result { background: #eaf3ff; border: 1px solid #bfdbfe;
        border-radius: 18px; padding: 24px; margin-top: 16px; }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button {
        background: #3b5668; color: white; border: 1px solid #3b5668;
        border-radius: 10px; font-weight: 700; min-height: 2.8rem;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button:hover { background: #1d4ed8; color: white; }
    h1, h2, h3 { color: #5d84a1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Model / dataset helpers -----------------------------
def find_model_file():
    """Find the expected model first, then another supported model file."""
    if MODEL_PATH.is_file():
        return MODEL_PATH

    candidates = []
    for folder in (MODEL_DIR, BASE_DIR):
        if folder.is_dir():
            for pattern in ("*.joblib", "*.pkl", "*.pickle"):
                candidates.extend(folder.glob(pattern))

    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None

    candidates.sort(key=lambda path: ("svc" not in path.name.lower(), "model" not in path.name.lower(), path.name.lower()))
    return candidates[0]


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model(path_string):
    # Load only model files you created or trust.
    return joblib.load(path_string)


def find_dataset_file():
    if DATASET_PATH.is_file():
        return DATASET_PATH
    for folder in (DATA_DIR, BASE_DIR):
        if folder.is_dir():
            for pattern in ("train*.csv", "*.csv"):
                for path in folder.glob(pattern):
                    if path.name.lower() != "test.csv":
                        return path
    return None


@st.cache_data(show_spinner=False)
def read_dataset(path_string):
    return pd.read_csv(path_string)


def get_feature_names(estimator):
    """Use the model's learned feature names when available."""
    names = getattr(estimator, "feature_names_in_", None)

    # A pipeline may expose feature names on the final estimator.
    if names is None and hasattr(estimator, "steps") and estimator.steps:
        names = getattr(estimator.steps[-1][1], "feature_names_in_", None)

    if names is not None:
        return [str(name) for name in names]

    # Fallback for the standard Kaggle mobile price dataset.
    return STANDARD_FEATURES.copy()


def get_reference_data():
    dataset_file = find_dataset_file()
    if dataset_file is None:
        return None, None
    try:
        df = read_dataset(str(dataset_file))
        return df, dataset_file
    except Exception:
        return None, dataset_file


def get_defaults(feature_names, dataframe):
    defaults = {}
    for feature in feature_names:
        if dataframe is not None and feature in dataframe.columns:
            values = pd.to_numeric(dataframe[feature], errors="coerce").dropna()
            if not values.empty:
                defaults[feature] = float(values.median())
                continue
        defaults[feature] = float(FEATURE_INFO.get(feature, ("", 0, 100, 0, ""))[3])
    return defaults


def render_feature_input(feature, defaults, key_prefix="feature"):
    label, min_value, max_value, fallback, help_text = FEATURE_INFO.get(
        feature,
        (feature.replace("_", " ").title(), 0, 10000, 0, "Feature required by the trained model."),
    )
    default = defaults.get(feature, float(fallback))
    default = min(max(float(default), float(min_value)), float(max_value))

    if feature in BINARY_FEATURES:
        return int(
            st.selectbox(
                label,
                options=[0, 1],
                index=int(round(default)),
                format_func=lambda value: "Yes" if value == 1 else "No",
                help=help_text,
                key=f"{key_prefix}_{feature}",
            )
        )

    if feature in {"clock_speed", "m_dep"}:
        return float(
            st.number_input(
                label,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(default),
                step=0.1,
                help=help_text,
                key=f"{key_prefix}_{feature}",
            )
        )

    return int(
        st.number_input(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(default),
            step=1.0,
            help=help_text,
            key=f"{key_prefix}_{feature}",
        )
    )


def class_description(label):
    """Describe the canonical target classes without inventing currency prices."""
    mapping = {
        0: ("Class 0", "Lowest relative price category"),
        1: ("Class 1", "Lower-middle relative price category"),
        2: ("Class 2", "Upper-middle relative price category"),
        3: ("Class 3", "Highest relative price category"),
    }
    try:
        return mapping.get(int(label), (str(label), "Predicted class from the trained model"))
    except (TypeError, ValueError):
        return str(label), "Predicted class from the trained model"


def estimator_name(estimator):
    if hasattr(estimator, "steps") and estimator.steps:
        final_estimator = estimator.steps[-1][1]
        return f"Pipeline → {type(final_estimator).__name__}"
    return type(estimator).__name__


# ----------------------------- Load resources -----------------------------
model_path = find_model_file()
model = None
model_error = None

if model_path is None:
    model_error = (
        "No saved model was found. Put best_svc_model.pkl in the models/ folder."
    )
else:
    try:
        model = load_model(str(model_path))
    except Exception as exc:
        model_error = str(exc)

reference_df, dataset_path = get_reference_data()
if reference_df is not None and TARGET_COLUMN in reference_df.columns:
    reference_features_df = reference_df.drop(columns=[TARGET_COLUMN])
else:
    reference_features_df = reference_df

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 📱 MobileSense")
    st.caption("Smart Mobile Price Prediction")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Home", "Price Prediction", "About the Project"],
        index=0,
    )

    st.divider()
    st.caption("Machine Learning · Classification")

    if model is not None:
        st.success(f"Model loaded: {model_path.name}")
    else:
        st.warning("Model is not loaded")

# ----------------------------- Home page -----------------------------
if page == "Home":
    st.markdown(
        """
        <div class="ms-hero">
            <div class="ms-eyebrow">Smartphone intelligence</div>
            <div class="ms-title">📱 MobileSense</div>
            <p class="ms-subtitle">Smart Mobile Price Prediction Using Machine Learning</p>
            <p style="color:#405b7c;max-width:780px;margin-top:18px;">
                Explore how smartphone specifications relate to price categories.
                Enter a phone's technical features and use a trained classification
                model to estimate its relative price class.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col2, col3 = st.columns(2)
    with col2:
        st.markdown(
            '<div class="ms-card"><h3>📊 Smartphone features</h3>'
            '<p>Explore RAM, battery, display, camera, memory, and connectivity specifications.</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="ms-card"><h3>⚡ Fast results</h3>'
            '<p>Enter specifications and receive a predicted class in seconds.</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.info("Select **Price Prediction** from the sidebar to predict a mobile price class.")

    if model is None:
        st.error(f"Model loading issue: {model_error}")
        st.caption("Check the model file location and install the dependencies used when it was saved.")

# ----------------------------- Prediction page -----------------------------
elif page == "Price Prediction":
    st.markdown("## Price Prediction")
    st.write("Enter smartphone specifications to predict its relative price class.")

    if model is None:
        st.error(f"Model unavailable: {model_error}")
        st.markdown("Place your trained model in `models/best_svc_model.pkl` and restart the app.")
        st.stop()

    feature_names = get_feature_names(model)
    if not feature_names:
        st.error("Could not determine the model's required input features.")
        st.stop()

    defaults = get_defaults(feature_names, reference_features_df)

    # Highlight the most useful inputs first; collect all remaining required inputs too.
    main_features = [feature for feature in MAIN_FEATURES if feature in feature_names]
    remaining_features = [feature for feature in feature_names if feature not in main_features]
    input_values = {}

    with st.form("mobile_prediction_form"):
        st.markdown("### Key specifications")
        st.caption(
            "Complete the main specifications. Expand Advanced specifications "
            "to edit any additional features required by the model."
        )

        groups = [
            ("Performance & Memory", ["ram", "int_memory", "n_cores"]),
            ("Battery & Display", ["battery_power", "px_height", "px_width"]),
            ("Camera", ["pc", "fc"]),
        ]

        for group_title, group_features in groups:
            selected = [feature for feature in group_features if feature in main_features]
            if selected:
                st.markdown(f"**{group_title}**")
                cols = st.columns(min(3, len(selected)))
                for index, feature in enumerate(selected):
                    with cols[index]:
                        input_values[feature] = render_feature_input(feature, defaults)

        ungrouped_main = [feature for feature in main_features if feature not in input_values]
        if ungrouped_main:
            cols = st.columns(min(3, len(ungrouped_main)))
            for index, feature in enumerate(ungrouped_main):
                with cols[index]:
                    input_values[feature] = render_feature_input(feature, defaults)

        with st.expander("Advanced specifications"):
            st.caption(
                "All additional features below are passed to the model. "
                "Defaults use the dataset median when that column is available."
            )
            for start in range(0, len(remaining_features), 3):
                row_features = remaining_features[start:start + 3]
                cols = st.columns(len(row_features))
                for index, feature in enumerate(row_features):
                    with cols[index]:
                        input_values[feature] = render_feature_input(
                            feature, defaults, key_prefix="advanced"
                        )

        submitted = st.form_submit_button(
            "🔎 Predict Mobile Price",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            # Ensure the exact expected feature order.
            input_df = pd.DataFrame(
                [[input_values[feature] for feature in feature_names]],
                columns=feature_names,
            )

            prediction = model.predict(input_df)[0]
            title, description = class_description(prediction)

            st.markdown(
                f"""
                <div class="ms-result">
                    <div class="ms-eyebrow">Prediction complete</div>
                    <h2 style="margin:8px 0;color:#12305a;">{title}</h2>
                    <p style="margin:0;color:#405b7c;">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "This result is a price category, not an exact retail price. "
                "Class meanings assume the standard Mobile Price Classification dataset."
            )

            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    final_estimator = (
                        model.steps[-1][1]
                        if hasattr(model, "steps") and model.steps
                        else model
                    )
                    classes = getattr(
                        final_estimator, "classes_", list(range(len(probabilities)))
                    )
                    probability_df = pd.DataFrame(
                        {
                            "Class": [str(value) for value in classes],
                            "Estimated probability": probabilities,
                        }
                    ).set_index("Class")

                    st.markdown("### Model-estimated class probabilities")
                    st.bar_chart(probability_df)
                    st.caption(
                        "These probabilities are model estimates, not guarantees of correctness."
                    )
                except Exception:
                    st.info("Class probabilities are not available for this model.")

            with st.expander("Review submitted model inputs"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(
                "Prediction failed. The model may require a different feature schema "
                "or preprocessing pipeline than the app currently supplies."
            )
            st.code(str(exc))

# ----------------------------- About page -----------------------------
else:
    st.markdown(
        """
        <div class="ms-hero">
            <div class="ms-eyebrow">About the project</div>
            <div class="ms-title" style="font-size:2.2rem;">MobileSense</div>
            <p class="ms-subtitle">
                Smart Mobile Price Prediction Using Machine Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project objective")
    st.write(
        "MobileSense uses a trained classification model to estimate a smartphone's "
        "relative price category from its technical specifications."
    )

    st.markdown("### Dataset")
    if dataset_path is not None and reference_df is not None:
        st.write(f"Dataset file detected: `{dataset_path.name}`")
        st.write(f"Rows: **{len(reference_df):,}** · Columns: **{len(reference_df.columns):,}**")
        if TARGET_COLUMN in reference_df.columns:
            st.write(f"Target column: `{TARGET_COLUMN}`")
            st.write("Target class values found in the dataset:")
            st.write(sorted(reference_df[TARGET_COLUMN].dropna().unique().tolist(), key=str))
    else:
        st.write(
            "The app did not find a readable CSV dataset. Put your training CSV in `data/` "
            "to display dataset details and derive default input values."
        )

    st.markdown("### Model")
    if model is not None:
        st.write(f"Model file: `{model_path.name}`")
        st.write(f"Detected estimator: **{estimator_name(model)}**")
        feature_names = get_feature_names(model)
        st.write(f"Detected input feature count: **{len(feature_names)}**")
        with st.expander("View expected input feature names"):
            st.code(", ".join(feature_names))
    else:
        st.error(f"Model is unavailable: {model_error}")

    st.markdown("### How it works")
    st.write(
        "1. Enter smartphone specifications. "
        "2. The app arranges them in the expected feature order. "
        "3. The saved model predicts a class. "
        "4. If supported, estimated class probabilities are displayed."
    )
    st.caption(
        "This application does not retrain the model. Make sure the saved file contains "
        "the same preprocessing used during model training. Pickle/joblib files should "
        "only be loaded if you trust their source."
    )
