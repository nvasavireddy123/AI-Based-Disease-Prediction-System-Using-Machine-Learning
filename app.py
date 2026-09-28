import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🩺 AI Disease Prediction System")

st.write(
    "An educational machine learning project that predicts "
    "a disease category based on selected symptoms."
)

st.warning(
    "⚠️ Educational project only. This system does not provide "
    "a medical diagnosis. Consult a qualified healthcare professional "
    "for medical advice."
)

# =========================================================
# DATASET
# =========================================================

data = {
    "Fever": [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0
    ],

    "Cough": [
        1, 1, 1, 0, 1,
        1, 1, 0, 1, 0,
        1, 0, 1, 1, 0,
        0, 1, 0, 1, 0
    ],

    "Headache": [
        1, 0, 1, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 1,
        1, 0, 1, 0, 1
    ],

    "Fatigue": [
        1, 1, 0, 1, 1,
        1, 0, 1, 0, 1,
        1, 0, 1, 1, 0,
        0, 1, 0, 1, 0
    ],

    "Sore Throat": [
        1, 1, 1, 0, 1,
        0, 0, 0, 1, 0,
        1, 0, 1, 0, 0,
        0, 1, 0, 1, 0
    ],

    "Runny Nose": [
        1, 0, 1, 0, 1,
        1, 1, 0, 1, 0,
        0, 1, 1, 0, 1,
        1, 0, 1, 0, 1
    ],

    "Nausea": [
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 0, 1
    ],

    "Skin Rash": [
        0, 0, 1, 0, 0,
        1, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        0, 1, 0, 1, 0
    ],

    "Disease": [
        "Flu", "Flu", "Flu", "Flu", "Flu",
        "Common Cold", "Common Cold", "Common Cold",
        "Common Cold", "Common Cold",
        "Viral Infection", "Viral Infection",
        "Viral Infection", "Viral Infection", "Viral Infection",
        "Allergy", "Allergy", "Allergy", "Allergy", "Allergy"
    ]
}

df = pd.DataFrame(data)

# =========================================================
# MACHINE LEARNING
# =========================================================

X = df.drop("Disease", axis=1)
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# MODEL ACCURACY
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📚 Project Information")

st.sidebar.write("### Machine Learning")
st.sidebar.write("Decision Tree Classifier")

st.sidebar.write("### Input")
st.sidebar.write("8 different symptoms")

st.sidebar.write("### Output")
st.sidebar.write("Predicted disease category")

st.sidebar.write("### Model Accuracy")

st.sidebar.success(
    f"{accuracy * 100:.2f}%"
)

# =========================================================
# SYMPTOM SELECTION
# =========================================================

st.header("🔍 Select Symptoms")

st.write(
    "Please select the symptoms currently present."
)

col1, col2 = st.columns(2)

with col1:

    fever = st.checkbox("🌡️ Fever")

    cough = st.checkbox("😷 Cough")

    headache = st.checkbox("🤕 Headache")

    fatigue = st.checkbox("😴 Fatigue")


with col2:

    sore_throat = st.checkbox("🗣️ Sore Throat")

    runny_nose = st.checkbox("🤧 Runny Nose")

    nausea = st.checkbox("🤢 Nausea")

    skin_rash = st.checkbox("🔴 Skin Rash")

# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Disease",
    type="primary",
    use_container_width=True
)

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame(
        [[
            int(fever),
            int(cough),
            int(headache),
            int(fatigue),
            int(sore_throat),
            int(runny_nose),
            int(nausea),
            int(skin_rash)
        ]],
        columns=X.columns
    )

    # -----------------------------------------------------
    # CHECK SYMPTOMS
    # -----------------------------------------------------

    total_symptoms = input_data.sum(axis=1).iloc[0]

    if total_symptoms == 0:

        st.error(
            "❌ Please select at least one symptom."
        )

    else:

        # -------------------------------------------------
        # MAKE PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_

        result_df = pd.DataFrame({
            "Disease Category": classes,
            "Probability": probabilities
        })

        result_df = result_df.sort_values(
            by="Probability",
            ascending=False
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.header("🩺 Prediction Result")

        st.success(
            f"Predicted Disease Category: **{prediction}**"
        )

        # -------------------------------------------------
        # SELECTED SYMPTOMS
        # -------------------------------------------------

        st.subheader("📋 Selected Symptoms")

        selected = []

        if fever:
            selected.append("Fever")

        if cough:
            selected.append("Cough")

        if headache:
            selected.append("Headache")

        if fatigue:
            selected.append("Fatigue")

        if sore_throat:
            selected.append("Sore Throat")

        if runny_nose:
            selected.append("Runny Nose")

        if nausea:
            selected.append("Nausea")

        if skin_rash:
            selected.append("Skin Rash")

        for symptom in selected:
            st.write("✔️", symptom)

        # -------------------------------------------------
        # PROBABILITY TABLE
        # -------------------------------------------------

        st.subheader("📊 Model Output")

        display_df = result_df.copy()

        display_df["Probability"] = (
            display_df["Probability"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # DISCLAIMER
        # -------------------------------------------------

        st.info(
            "⚠️ The prediction is based on the small educational "
            "dataset used in this demonstration. It should not be "
            "used to diagnose or treat a medical condition."
        )

# =========================================================
# ABOUT PROJECT
# =========================================================

st.divider()

st.header("📖 About the Project")

st.write(
    """
    This project demonstrates how Machine Learning can be used
    for classification.

    The model learns patterns between symptoms and disease
    categories from the training dataset.

    Workflow:

    Data → Preprocessing → Training → Testing → Prediction
    """
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Disease Prediction System | Academic Machine Learning Project"
)