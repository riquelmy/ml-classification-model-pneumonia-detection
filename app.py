import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub
import random

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection CNN",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = 150
LABELS = ["PNEUMONIA", "NORMAL"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_cnn_model.keras")


# ─── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_data_dir():
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    for root, dirs, _ in os.walk(path):
        if "chest_xray" in dirs:
            return os.path.join(root, "chest_xray")
    return path


@st.cache_data
def load_dataset_split(data_dir, split):
    x, y = [], []
    split_dir = os.path.join(data_dir, split)
    for label in LABELS:
        label_dir = os.path.join(split_dir, label)
        class_num = LABELS.index(label)
        for img_name in os.listdir(label_dir):
            try:
                img = cv2.imread(os.path.join(label_dir, img_name), cv2.IMREAD_GRAYSCALE)
                resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                x.append(resized)
                y.append(class_num)
            except Exception:
                pass
    x = np.array(x) / 255.0
    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    return x, y


@st.cache_data
def get_dataset_counts(data_dir):
    counts = {}
    for split in ["train", "test", "val"]:
        counts[split] = {}
        for label in LABELS:
            label_dir = os.path.join(data_dir, split, label)
            counts[split][label] = len(os.listdir(label_dir))
    return counts


@st.cache_data
def compute_test_metrics(_model, data_dir):
    x_test, y_test = load_dataset_split(data_dir, "test")
    loss, accuracy = _model.evaluate(x_test, y_test, verbose=0)
    preds = (_model.predict(x_test, verbose=0) > 0.5).astype("int32").flatten()
    report = classification_report(
        y_test, preds,
        target_names=["Pneumonia (0)", "Normal (1)"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, preds)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "predictions": preds,
        "y_test": y_test,
        "x_test": x_test,
    }


# ─── Helper: Predict a single image ────────────────────────────────────────────
def predict_single(model, img_array_gray):
    resized = cv2.resize(img_array_gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prob = model.predict(reshaped, verbose=0)[0][0]
    label = "NORMAL" if prob >= 0.5 else "PNEUMONIA"
    confidence = prob if label == "NORMAL" else 1 - prob
    return label, float(confidence)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/lungs.png",
    width=80,
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "🧠 Model Architecture",
        "📈 Training Results",
        "🔬 Test Evaluation",
        "🩻 Predict Your X-Ray",
        "🖼️ Sample Predictions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit + TensorFlow/Keras")


# ─── Load model and data ───────────────────────────────────────────────────────
model = load_trained_model()
data_dir = get_data_dir()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🫁 Pneumonia Detection using CNN")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            ### What is Pneumonia?
            **Pneumonia** is an inflammatory condition of the lung affecting primarily 
            the small air sacs known as **alveoli**. Symptoms typically include some 
            combination of productive or dry cough, chest pain, fever and difficulty breathing.

            Pneumonia is usually caused by infection with **viruses or bacteria** and less 
            commonly by other microorganisms. Risk factors include cystic fibrosis, COPD, 
            asthma, diabetes, heart failure, and a weak immune system.

            **Chest X-rays** are one of the primary tools for diagnosing pneumonia. This 
            project uses a **Convolutional Neural Network (CNN)** to automatically classify 
            chest X-ray images as either **Normal** or **Pneumonia**.

            ### Project Highlights
            - **5,863 chest X-ray images** from pediatric patients (ages 1-5)
            - **5-layer CNN** with BatchNormalization and Dropout
            - **Data augmentation** to handle class imbalance
            - **~91%+ test accuracy** on unseen data
            """
        )
    with col2:
        st.markdown("### Quick Stats")
        metrics = compute_test_metrics(model, data_dir)
        st.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
        st.metric("Test Loss", f"{metrics['loss']:.4f}")
        st.metric("Total Parameters", "1,246,401")
        st.metric("Trainable Parameters", "1,245,313")

    st.markdown("---")
    st.info(
        "👈 Use the **sidebar** to navigate through the different sections of this presentation."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("---")

    counts = get_dataset_counts(data_dir)

    # Dataset split overview
    st.subheader("Dataset Split Overview")
    col1, col2, col3 = st.columns(3)
    for col, split in zip([col1, col2, col3], ["train", "test", "val"]):
        total = counts[split]["PNEUMONIA"] + counts[split]["NORMAL"]
        with col:
            st.markdown(f"#### {split.capitalize()} Set")
            st.metric("Total Images", total)
            st.metric("Pneumonia", counts[split]["PNEUMONIA"])
            st.metric("Normal", counts[split]["NORMAL"])

    st.markdown("---")

    # Class distribution chart
    st.subheader("Class Distribution")
    fig = go.Figure()
    for label in LABELS:
        fig.add_trace(go.Bar(
            name=label,
            x=["Train", "Test", "Val"],
            y=[counts[s][label] for s in ["train", "test", "val"]],
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Split",
        yaxis_title="Number of Images",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Sample images
    st.subheader("Sample X-Ray Images")
    sample_label = st.selectbox("Select class", LABELS)
    sample_split = st.selectbox("Select split", ["train", "test", "val"])
    sample_dir = os.path.join(data_dir, sample_split, sample_label)
    sample_files = os.listdir(sample_dir)
    random.seed(42)
    chosen = random.sample(sample_files, min(4, len(sample_files)))

    cols = st.columns(4)
    for i, fname in enumerate(chosen):
        img = cv2.imread(os.path.join(sample_dir, fname), cv2.IMREAD_GRAYSCALE)
        with cols[i]:
            st.image(img, caption=f"{sample_label}", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        **Dataset Source:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
        Chest X-ray images (anterior-posterior) were selected from retrospective cohorts 
        of pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Architecture
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Architecture":
    st.title("🧠 CNN Model Architecture")
    st.markdown("---")

    st.markdown(
        """
        The model is a **Sequential CNN** with 5 convolutional blocks followed by 
        fully connected layers. Key techniques used:
        - **Batch Normalization** after each Conv layer for stable training
        - **Dropout** (10-20%) to prevent overfitting
        - **MaxPooling** to reduce spatial dimensions
        - **RMSprop** optimizer with **ReduceLROnPlateau** callback
        """
    )

    # Architecture diagram
    layers_data = [
        ("Input", "150×150×1", "-"),
        ("Conv2D (32 filters, 3×3)", "150×150×32", "320"),
        ("BatchNorm + MaxPool", "75×75×32", "128"),
        ("Conv2D (64 filters, 3×3)", "75×75×64", "18,496"),
        ("Dropout(0.1) + BatchNorm + MaxPool", "38×38×64", "256"),
        ("Conv2D (64 filters, 3×3)", "38×38×64", "36,928"),
        ("BatchNorm + MaxPool", "19×19×64", "256"),
        ("Conv2D (128 filters, 3×3)", "19×19×128", "73,856"),
        ("Dropout(0.2) + BatchNorm + MaxPool", "10×10×128", "512"),
        ("Conv2D (256 filters, 3×3)", "10×10×256", "295,168"),
        ("Dropout(0.2) + BatchNorm + MaxPool", "5×5×256", "1,024"),
        ("Flatten", "6,400", "0"),
        ("Dense (128, ReLU)", "128", "819,328"),
        ("Dropout(0.2)", "128", "0"),
        ("Dense (1, Sigmoid)", "1", "129"),
    ]

    st.subheader("Layer-by-Layer Breakdown")
    header_cols = st.columns([3, 2, 1])
    header_cols[0].markdown("**Layer**")
    header_cols[1].markdown("**Output Shape**")
    header_cols[2].markdown("**Parameters**")

    for layer_name, shape, params in layers_data:
        cols = st.columns([3, 2, 1])
        cols[0].markdown(f"`{layer_name}`")
        cols[1].markdown(shape)
        cols[2].markdown(params)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Parameters", "1,246,401")
    col2.metric("Trainable", "1,245,313")
    col3.metric("Non-Trainable", "1,088")

    st.markdown("---")
    st.subheader("Data Augmentation Strategy")
    aug_data = {
        "Technique": [
            "Rotation", "Zoom", "Width Shift",
            "Height Shift", "Horizontal Flip",
        ],
        "Value": ["±30°", "20%", "10%", "10%", "Yes"],
        "Purpose": [
            "Simulate varied patient positioning",
            "Handle distance variations",
            "Account for lateral shifts",
            "Account for vertical shifts",
            "Increase data variety",
        ],
    }
    st.table(aug_data)

    st.markdown(
        """
        **Why Data Augmentation?**  
        The training set is imbalanced (3,875 Pneumonia vs 1,341 Normal). Augmentation 
        artificially expands the dataset by applying random transformations, helping the 
        model generalize better and reducing overfitting.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Training Results
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training Results":
    st.title("📈 Training Results")
    st.markdown("---")

    st.markdown(
        """
        The model was trained for **12 epochs** with the **RMSprop** optimizer 
        and **ReduceLROnPlateau** learning rate scheduler (patience=2, factor=0.3).
        """
    )

    # Training history (from actual training run)
    train_acc = [0.8418, 0.9005, 0.9250, 0.9511, 0.9551, 0.9609, 0.9632, 0.9647, 0.9647, 0.9670, 0.9703, 0.9682]
    val_acc = [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.6875, 0.5000, 0.7500, 0.5625, 0.7500, 0.5000, 0.6250]
    train_loss = [0.4730, 0.2531, 0.2100, 0.1470, 0.1332, 0.1131, 0.1055, 0.1054, 0.0999, 0.0930, 0.0951, 0.0968]
    val_loss = [29.9704, 37.1825, 56.6157, 23.3892, 33.6988, 0.7274, 3.1937, 0.5703, 8.2574, 0.5179, 2.0359, 1.4719]
    epochs = list(range(1, 13))

    col1, col2 = st.columns(2)

    with col1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=train_acc, mode="lines+markers",
            name="Training Accuracy", line=dict(color="#2ecc71", width=2),
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=val_acc, mode="lines+markers",
            name="Validation Accuracy", line=dict(color="#e74c3c", width=2),
        ))
        fig_acc.update_layout(
            title="Accuracy over Epochs",
            xaxis_title="Epoch", yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1.05]),
            height=400,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=train_loss, mode="lines+markers",
            name="Training Loss", line=dict(color="#2ecc71", width=2),
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=val_loss, mode="lines+markers",
            name="Validation Loss", line=dict(color="#e74c3c", width=2),
        ))
        fig_loss.update_layout(
            title="Loss over Epochs",
            xaxis_title="Epoch", yaxis_title="Loss",
            height=400,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("---")
    st.subheader("Learning Rate Schedule")

    lr_data = {
        "Epoch": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Learning Rate": [
            "1.0e-3", "1.0e-3", "3.0e-4 ↓", "3.0e-4",
            "9.0e-5 ↓", "9.0e-5", "9.0e-5", "9.0e-5",
            "9.0e-5", "2.7e-5 ↓", "2.7e-5", "8.1e-6 ↓",
        ],
    }
    st.table(lr_data)

    st.info(
        "⚠️ **Note:** The validation set is very small (16 images), which causes erratic "
        "validation metrics. The test set (624 images) provides a much more reliable evaluation."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Test Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Test Evaluation":
    st.title("🔬 Test Set Evaluation")
    st.markdown("---")

    metrics = compute_test_metrics(model, data_dir)

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("Loss", f"{metrics['loss']:.4f}")
    col3.metric("Pneumonia F1", f"{metrics['report']['Pneumonia (0)']['f1-score']:.2%}")
    col4.metric("Normal F1", f"{metrics['report']['Normal (1)']['f1-score']:.2%}")

    st.markdown("---")

    # Classification report
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Classification Report")
        report = metrics["report"]
        report_rows = []
        for cls in ["Pneumonia (0)", "Normal (1)"]:
            report_rows.append({
                "Class": cls,
                "Precision": f"{report[cls]['precision']:.2%}",
                "Recall": f"{report[cls]['recall']:.2%}",
                "F1-Score": f"{report[cls]['f1-score']:.2%}",
                "Support": int(report[cls]["support"]),
            })
        report_rows.append({
            "Class": "**Weighted Avg**",
            "Precision": f"{report['weighted avg']['precision']:.2%}",
            "Recall": f"{report['weighted avg']['recall']:.2%}",
            "F1-Score": f"{report['weighted avg']['f1-score']:.2%}",
            "Support": int(report['weighted avg']['support']),
        })
        st.table(report_rows)

    with col_right:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["PNEUMONIA", "NORMAL"],
            y=["PNEUMONIA", "NORMAL"],
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="equal",
        )
        fig_cm.update_layout(height=400, width=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # Correct / Incorrect predictions
    st.subheader("Prediction Examples from Test Set")
    x_test = metrics["x_test"]
    y_test = metrics["y_test"]
    preds = metrics["predictions"]
    correct_idx = np.nonzero(preds == y_test)[0]
    incorrect_idx = np.nonzero(preds != y_test)[0]

    tab1, tab2 = st.tabs(["✅ Correct Predictions", "❌ Incorrect Predictions"])

    with tab1:
        cols = st.columns(6)
        for i, idx in enumerate(correct_idx[:6]):
            with cols[i]:
                img = (x_test[idx].reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
                st.image(img, caption=f"Pred: {LABELS[preds[idx]]}\nTrue: {LABELS[y_test[idx]]}", use_container_width=True)

    with tab2:
        if len(incorrect_idx) > 0:
            cols = st.columns(min(6, len(incorrect_idx)))
            for i, idx in enumerate(incorrect_idx[:6]):
                with cols[i]:
                    img = (x_test[idx].reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
                    st.image(img, caption=f"Pred: {LABELS[preds[idx]]}\nTrue: {LABELS[y_test[idx]]}", use_container_width=True)
        else:
            st.success("No incorrect predictions!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict Your X-Ray
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🩻 Predict Your X-Ray":
    st.title("🩻 Upload & Predict")
    st.markdown("---")
    st.markdown(
        "Upload a **chest X-ray image** (JPEG/PNG) and the model will predict whether "
        "it shows signs of **Pneumonia** or is **Normal**."
    )

    uploaded = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a grayscale or color chest X-ray image",
    )

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Uploaded Image")
            display_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_container_width=True)

        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing X-ray..."):
                label, confidence = predict_single(model, img_gray)

            if label == "PNEUMONIA":
                st.error(f"### ⚠️ {label}")
            else:
                st.success(f"### ✅ {label}")

            st.metric("Confidence", f"{confidence:.1%}")

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={"text": f"Prediction: {label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if label == "PNEUMONIA" else "#2ecc71"},
                    "steps": [
                        {"range": [0, 50], "color": "#fadbd8"},
                        {"range": [50, 75], "color": "#fdebd0"},
                        {"range": [75, 100], "color": "#d5f5e3"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.warning(
            "⚕️ **Disclaimer:** This tool is for educational purposes only. "
            "It should NOT be used as a substitute for professional medical diagnosis."
        )
    else:
        st.info("👆 Upload a chest X-ray image to get started.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Sample Predictions
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Sample Predictions":
    st.title("🖼️ Sample Predictions from Validation Set")
    st.markdown("---")

    num_samples = st.slider("Number of samples", 4, 16, 8, step=4)

    if st.button("🔄 Generate New Random Samples", type="primary"):
        st.session_state["sample_seed"] = random.randint(0, 99999)

    seed = st.session_state.get("sample_seed", 42)

    val_dir = os.path.join(data_dir, "val")
    val_images = []
    for label in LABELS:
        label_dir = os.path.join(val_dir, label)
        for img_name in os.listdir(label_dir):
            val_images.append((os.path.join(label_dir, img_name), label))

    rng = random.Random(seed)
    rng.shuffle(val_images)
    samples = val_images[:num_samples]

    cols_per_row = 4
    for row_start in range(0, len(samples), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, (img_path, true_label) in enumerate(samples[row_start:row_start + cols_per_row]):
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            predicted_label, confidence = predict_single(model, img_gray)
            is_correct = predicted_label == true_label

            with cols[i]:
                st.image(img_gray, use_container_width=True)
                if is_correct:
                    st.success(f"✅ Pred: **{predicted_label}** ({confidence:.0%})")
                else:
                    st.error(f"❌ Pred: **{predicted_label}** ({confidence:.0%})")
                st.caption(f"True: {true_label}")

    st.markdown("---")
    total = len(samples)
    correct_count = sum(
        1 for img_path, true_label in samples
        if predict_single(model, cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))[0] == true_label
    )
    st.metric("Sample Accuracy", f"{correct_count}/{total} ({correct_count/total:.0%})")

    st.warning(
        "⚠️ **Note:** The validation set contains only **16 images** (8 Pneumonia + 8 Normal). "
        "This very small sample size means results here may not be statistically representative. "
        "Refer to the **Test Evaluation** page (624 images) for a more reliable assessment of model performance."
    )
