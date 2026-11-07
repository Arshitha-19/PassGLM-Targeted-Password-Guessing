import gc
import json
import re

import pandas as pd
import streamlit as st
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

import config


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PassGLM Research Demo",
    page_icon="🔐",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 18px;
            color: #777777;
            margin-top: 0;
        }

        .info-card {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #dddddd;
            margin-bottom: 12px;
        }

        .warning-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #fff4e5;
            border: 1px solid #f0c36d;
        }

        .success-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #eaf7ea;
            border: 1px solid #8bc48b;
        }

        .project-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #dddddd;
            background-color: #fafafa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">PassGLM Research Demo</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Targeted Password Prediction using Synthetic Research Data'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

st.warning(
    "Research demonstration only. This application uses synthetic/de-identified "
    "research data and locally trained research models. Do not use it with real "
    "credentials or real personal information."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_prediction(text):
    """
    Clean a generated model prediction.
    This follows the same cleaning approach used by evaluate.py.
    """

    text = text.strip()

    if not text:
        return ""

    text = text.splitlines()[0].strip()

    text = re.split(r"[\s,;]+", text)[0]

    return text.strip("`'\"")


def clear_model_from_memory():
    """
    Release the currently loaded model from GPU/CPU memory.
    """

    if "model" in st.session_state:
        try:
            del st.session_state.model
        except Exception:
            pass

    if "tokenizer" in st.session_state:
        try:
            del st.session_state.tokenizer
        except Exception:
            pass

    st.session_state.loaded_mode = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_mode_key(mode):
    """
    Convert UI mode name to project mode name.
    """

    mode_mapping = {
        "PII": "pii",
        "Old Password": "old",
        "Combined": "combined",
    }

    return mode_mapping[mode]


# ============================================================
# MODEL LOADING
# ============================================================


def load_model(mode):
    """
    Load the Qwen base model and attach the selected LoRA adapter.
    Uses 4-bit inference on CUDA GPUs and FP32 inference on CPU.
    """

    adapter_dir = (
        config.CHECKPOINT_DIR
        / mode
        / "final"
    )

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found:\n{adapter_dir}"
        )

    current_mode = st.session_state.get("loaded_mode")

    if current_mode is not None and current_mode != mode:
        clear_model_from_memory()

    if (
        st.session_state.get("loaded_mode") == mode
        and "model" in st.session_state
        and "tokenizer" in st.session_state
    ):
        return (
            st.session_state.tokenizer,
            st.session_state.model,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=quant_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )

    else:

        st.info("Running model on CPU. Inference may be slower.")

        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float32,
        )

        base_model.to("cpu")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )

    model.eval()

    st.session_state.tokenizer = tokenizer
    st.session_state.model = model
    st.session_state.loaded_mode = mode

    return tokenizer, model

# ============================================================
# TEST DATA LOADING
# ============================================================

@st.cache_data
def load_test_data(mode):
    """
    Load the processed synthetic test data for a given mode.
    """

    test_file = (
        config.PROCESSED_DIR
        / mode
        / "test.jsonl"
    )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test data not found:\n{test_file}"
        )

    df = pd.read_json(
        test_file,
        lines=True,
    )

    return df


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    tokenizer,
    model,
    prompt,
    num_candidates,
):
    """
    Generate ranked candidate predictions using beam search.

    This mirrors the generation approach in evaluate.py.
    """

    device = next(
        model.parameters()
    ).device

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_LENGTH,
    ).to(device)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # --------------------------------------------------------
    # Remove prompt tokens from generated sequences.
    # --------------------------------------------------------

    prompt_len = inputs["input_ids"].shape[1]

    candidates = []

    for sequence in outputs:

        generated_text = tokenizer.decode(
            sequence[prompt_len:],
            skip_special_tokens=True,
        )

        prediction = clean_prediction(
            generated_text
        )

        if (
            prediction
            and prediction not in candidates
        ):
            candidates.append(
                prediction
            )

    return candidates


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "loaded_mode" not in st.session_state:
    st.session_state.loaded_mode = None

if "model" not in st.session_state:
    st.session_state.model = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Experiment Settings")

    mode = st.selectbox(
        "Input Mode",
        [
            "PII",
            "Old Password",
            "Combined",
        ],
    )

    mode_key = get_mode_key(mode)

    st.markdown("---")

    st.subheader("Generation")

    beams = st.selectbox(
        "Number of Candidates",
        [
            5,
            10,
            20,
        ],
        index=2,
    )

    st.markdown("---")

    st.subheader("Model")

    st.write(
        "**Base Model:** "
        "Qwen2.5-0.5B-Instruct"
    )

    st.write(
        "**Fine-tuning:** LoRA"
    )

    if torch.cuda.is_available():
        st.write("**Quantization:** 4-bit NF4")
    else:
        st.write("**Execution:** CPU (FP32)")

    if torch.cuda.is_available():
        st.success("CUDA GPU detected")
        gpu_name = torch.cuda.get_device_name(0)
        st.caption(f"GPU: {gpu_name}")
    else:
         st.info("CPU mode enabled (Intel Graphics)")

    st.markdown("---")

    st.caption(
        "The demo uses already-trained LoRA checkpoints. "
        "No training is performed inside Streamlit."
    )


# ============================================================
# TABS
# ============================================================

tab_demo, tab_results, tab_architecture, tab_about = st.tabs(
    [
        "Prediction Demo",
        "Experimental Results",
        "Architecture",
        "About",
    ]
)


# ============================================================
# PREDICTION DEMO TAB
# ============================================================

with tab_demo:

    st.header(
        f"{mode} Prediction"
    )

    st.write(
        "Select a synthetic test record and generate ranked "
        "candidate predictions using the trained model."
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    try:

        test_df = load_test_data(
            mode_key
        )

        if len(test_df) == 0:

            st.error(
                "No synthetic test records are available."
            )

        else:

            # ------------------------------------------------
            # Select synthetic test record
            # ------------------------------------------------

            max_index = len(test_df) - 1

            selected_index = st.number_input(
                "Synthetic Test Record",
                min_value=0,
                max_value=max_index,
                value=0,
                step=1,
            )

            selected_row = test_df.iloc[
                selected_index
            ]

            # ------------------------------------------------
            # Input information
            # ------------------------------------------------

            st.subheader(
                "Synthetic Input Information"
            )

            display_fields = [
                "user_id",
                "name",
                "birthday",
                "phone",
                "email",
                "old_password",
            ]

            available_fields = [
                field
                for field in display_fields
                if field in selected_row.index
            ]

            if available_fields:

                input_data = {}

                for field in available_fields:

                    value = selected_row[field]

                    if pd.isna(value):
                        value = ""

                    input_data[field] = value

                input_table = (
                    pd.DataFrame(
                        [input_data]
                    )
                    .T
                    .rename(
                        columns={
                            0: "Synthetic Value"
                        }
                    )
                )

                st.dataframe(
                    input_table,
                    use_container_width=True,
                )

            else:

                st.info(
                    "No displayable input fields were found "
                    "for this synthetic record."
                )

            st.info(
                "The model prompt is taken directly from the "
                "processed synthetic test set."
            )

            # ------------------------------------------------
            # Generate button
            # ------------------------------------------------

            if st.button(
                "Generate Candidates",
                type="primary",
                use_container_width=True,
            ):

                try:

                    with st.spinner(
                        "Loading model and generating candidates..."
                    ):

                        tokenizer, model = load_model(
                            mode_key
                        )

                        candidates = generate_candidates(
                            tokenizer,
                            model,
                            selected_row["prompt"],
                            beams,
                        )

                    # ------------------------------------------------
                    # Generation completed
                    # ------------------------------------------------

                    st.success(
                        f"Generated {len(candidates)} unique candidates."
                    )

                    # ------------------------------------------------
                    # Candidate table
                    # ------------------------------------------------

                    st.subheader(
                        "Ranked Candidate Predictions"
                    )

                    candidate_rows = []

                    target = str(
                        selected_row["target"]
                    )

                    target_rank = None

                    for rank, candidate in enumerate(
                        candidates,
                        start=1,
                    ):

                        is_target = (
                            candidate == target
                        )

                        if is_target:
                            target_rank = rank

                        candidate_rows.append(
                            {
                                "Rank": rank,
                                "Candidate": candidate,
                                "Match": (
                                    "YES"
                                    if is_target
                                    else ""
                                ),
                            }
                        )

                    result_df = pd.DataFrame(
                        candidate_rows
                    )

                    if len(result_df) > 0:

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:

                        st.warning(
                            "The model did not produce "
                            "any valid unique candidates."
                        )

                    # ------------------------------------------------
                    # Prediction metrics
                    # ------------------------------------------------

                    st.subheader(
                        "Prediction Evaluation"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Unique Candidates",
                            len(candidates),
                        )

                    with col2:

                        if target_rank is not None:

                            st.metric(
                                "Target Rank",
                                target_rank,
                            )

                        else:

                            st.metric(
                                "Target Rank",
                                "Not Found",
                            )

                    with col3:

                        if target_rank is not None:

                            reciprocal_rank = (
                                1.0 / target_rank
                            )

                            st.metric(
                                "Reciprocal Rank",
                                f"{reciprocal_rank:.4f}",
                            )

                        else:

                            st.metric(
                                "Reciprocal Rank",
                                "0.0000",
                            )

                    # ------------------------------------------------
                    # Ground truth
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="success-box">'
                        "<b>Synthetic Ground Truth:</b> "
                        + target
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                except Exception as e:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(e)

    except Exception as e:

        st.error(
            "Could not load the synthetic test data."
        )

        st.exception(e)


# ============================================================
# EXPERIMENTAL RESULTS TAB
# ============================================================

with tab_results:

    st.header(
        "Experimental Results"
    )

    st.write(
        "These results are loaded directly from the metrics "
        "generated by the project's evaluation pipeline."
    )

    result_data = {}

    # --------------------------------------------------------
    # Load saved metrics
    # --------------------------------------------------------

    for result_mode in [
        "pii",
        "old",
        "combined",
    ]:

        metrics_file = (
            config.METRICS_DIR
            / f"{result_mode}_top100.json"
        )

        if metrics_file.exists():

            try:

                with metrics_file.open(
                    "r",
                    encoding="utf-8",
                ) as f:

                    result_data[
                        result_mode
                    ] = json.load(f)

            except Exception as e:

                st.warning(
                    f"Could not read {metrics_file.name}: {e}"
                )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    if not result_data:

        st.warning(
            "No saved metrics were found in "
            f"{config.METRICS_DIR}"
        )

    else:

        table_rows = []

        for result_mode, metrics in result_data.items():

            table_rows.append(
                {
                    "Mode": result_mode.title(),
                    "Top-1": (
                        metrics.get(
                            "top_1",
                            0,
                        )
                        * 100
                    ),
                    "Top-5": (
                        metrics.get(
                            "top_5",
                            0,
                        )
                        * 100
                    ),
                    "Top-10": (
                        metrics.get(
                            "top_10",
                            0,
                        )
                        * 100
                    ),
                    "Top-50": (
                        metrics.get(
                            "top_50",
                            0,
                        )
                        * 100
                    ),
                    "Top-100": (
                        metrics.get(
                            "top_100",
                            0,
                        )
                        * 100
                    ),
                    "MRR": metrics.get(
                        "mrr",
                        0,
                    ),
                }
            )

        results_df = pd.DataFrame(
            table_rows
        )

        # ----------------------------------------------------
        # Results table
        # ----------------------------------------------------

        st.subheader(
            "Top-K and MRR Results"
        )

        formatted_df = results_df.copy()

        for column in [
            "Top-1",
            "Top-5",
            "Top-10",
            "Top-50",
            "Top-100",
        ]:

            formatted_df[column] = (
                formatted_df[column]
                .map(
                    lambda x: f"{x:.1f}%"
                )
            )

        formatted_df["MRR"] = (
            formatted_df["MRR"]
            .map(
                lambda x: f"{x:.4f}"
            )
        )

        st.dataframe(
            formatted_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Top-K chart
        # ----------------------------------------------------

        st.subheader(
            "Top-K Comparison"
        )

        chart_df = results_df.set_index(
            "Mode"
        )[
            [
                "Top-1",
                "Top-5",
                "Top-10",
                "Top-50",
                "Top-100",
            ]
        ]

        st.bar_chart(
            chart_df
        )

        # ----------------------------------------------------
        # MRR chart
        # ----------------------------------------------------

        st.subheader(
            "MRR Comparison"
        )

        mrr_df = results_df.set_index(
            "Mode"
        )[
            [
                "MRR"
            ]
        ]

        st.bar_chart(
            mrr_df
        )


# ============================================================
# ARCHITECTURE TAB
# ============================================================

with tab_architecture:

    st.header(
        "System Architecture"
    )

    st.markdown(
        """
        ### Research Pipeline

        **1. Synthetic Dataset**

        ↓

        **2. Data Preprocessing**

        ↓

        **3. Input Configuration**

        - PII
        - Old Password
        - Combined

        ↓

        **4. Prompt Construction**

        ↓

        **5. Qwen2.5-0.5B-Instruct**

        ↓

        **6. 4-bit NF4 Quantization + LoRA**

        ↓

        **7. Candidate Generation**

        ↓

        **8. Beam Search Ranking**

        ↓

        **9. Top-K Evaluation**

        ↓

        **10. MRR Evaluation**

        ↓

        **11. Candidate Predictions + Metrics**
        """
    )

    st.divider()

    st.subheader(
        "Model Configuration"
    )

    config_table = pd.DataFrame(
        {
            "Parameter": [
                "Base Model",
                "Dataset Size",
                "Training Epochs",
                "Maximum Sequence Length",
                "Batch Size",
                "Gradient Accumulation",
                "Learning Rate",
                "LoRA Rank",
                "LoRA Alpha",
                "LoRA Dropout",
                "4-bit Quantization",
                "Generation Beams",
                "Maximum New Tokens",
            ],
            "Value": [
                config.MODEL_NAME,
                f"{config.NUM_USERS:,}",
                config.NUM_EPOCHS,
                config.MAX_LENGTH,
                config.BATCH_SIZE,
                config.GRADIENT_ACCUMULATION_STEPS,
                config.LEARNING_RATE,
                config.LORA_R,
                config.LORA_ALPHA,
                config.LORA_DROPOUT,
                "Enabled",
                config.GENERATION_BEAMS,
                config.MAX_NEW_TOKENS,
            ],
        }
    )

    st.dataframe(
        config_table,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader(
        "Input Modes"
    )

    mode_table = pd.DataFrame(
        {
            "Mode": [
                "PII",
                "Old Password",
                "Combined",
            ],
            "Input": [
                "Synthetic PII information",
                "Synthetic previous password",
                "Synthetic PII + previous password",
            ],
        }
    )

    st.dataframe(
        mode_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.header(
        "About the Project"
    )

    st.markdown(
        """
        ## PASSGLM: AI-BASED TARGETED PASSWORD PREDICTION MODEL

        This project is an academic research implementation inspired by
        the PassGLM methodology.

        The project studies whether a language model can generate
        password-like candidate predictions when supplied with different
        forms of synthetic user information.

        ### Experimental Conditions

        **PII Mode**

        Uses synthetic personally identifiable information.

        **Old Password Mode**

        Uses a synthetic previous password.

        **Combined Mode**

        Uses both synthetic PII and a synthetic previous password.

        ### Model

        - Qwen2.5-0.5B-Instruct
        - LoRA fine-tuning
        - 4-bit NF4 quantization
        - Beam-search candidate generation

        ### Dataset

        - 10,000 synthetic records
        - 80% training data
        - 20% testing data
        - Separate experiments for PII, Old Password and Combined modes

        ### Evaluation

        The system evaluates:

        - Top-1 accuracy
        - Top-5 accuracy
        - Top-10 accuracy
        - Top-50 accuracy
        - Top-100 accuracy
        - Mean Reciprocal Rank (MRR)

        ### Ethical Scope

        This implementation uses synthetic/de-identified research data
        and is intended for academic experimentation and defensive
        security research.

        It should not be used with real credentials, real leaked
        password datasets, or real people's personal information.
        """
    )

    st.divider()

    st.subheader(
        "Project Environment"
    )

    environment_table = pd.DataFrame(
        {
            "Component": [
                "Programming Language",
                "Framework",
                "Base Model",
                "Fine-tuning",
                "Quantization",
                "Dataset",
                "Evaluation",
            ],
            "Technology": [
                "Python",
                "Streamlit",
                config.MODEL_NAME,
                "LoRA",
                "4-bit NF4",
                "Synthetic",
                "Top-K + MRR",
            ],
        }
    )

    st.dataframe(
        environment_table,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.caption(
        "KITS Warangal | CSE (AI & ML) | Academic Year 2026-27"
    )