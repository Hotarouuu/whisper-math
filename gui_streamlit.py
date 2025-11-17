import streamlit as st
import re
import numpy as np
from sympy import sympify, sqrt, N
from sympy.core.sympify import SympifyError
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import soundfile as sf
from io import BytesIO
import time

# ====== Loading the models ======


@st.cache_resource
def load_models():
    print("Loading models...")
    proc = AutoProcessor.from_pretrained(
        "manushya-ai/whisper-medium-finetuned")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "manushya-ai/whisper-medium-finetuned")
    print("Models loaded.")
    return proc, model


# ====== ASR settings ======
SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
DEVICE = 27                 # mic index if needed (use print_devices())
COMPUTE_TYPE = "int8"          # good on RPi5: "int8" or "int8_float16"
BEAM_SIZE = 5
TEMPERATURE = 0.0

# ====== INITIAL PROMPTS ======
INITIAL_PROMPT_ar = "Arabic math: صفر واحد اثنان اثنين ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة زائد ناقص ضرب قسمة جذر تربيعي يساوي نسبة مئة فاصلة نقطة أس تربيع تكعيب"
INITIAL_PROMPT_en = "English math: zero one two three four five six seven eight nine ten plus minus times multiply divide over square root equals percent point comma power squared cubed"

# ====== Text normalization (EN + AR → mathy string) ======
WORD_MAP = {
    # English digits
    r"\bzero\b": "0", r"\bone\b": "1", r"\btwo\b": "2", r"\bthree\b": "3", r"\bfour\b": "4",
    r"\bfive\b": "5", r"\bsix\b": "6", r"\bseven\b": "7", r"\beight\b": "8", r"\bnine\b": "9",
    r"\bten\b": "10",

    # English teens
    r"\beleven\b": "11", r"\btwelve\b": "12", r"\bthirteen\b": "13", r"\bfourteen\b": "14",
    r"\bfifteen\b": "15", r"\bsixteen\b": "16", r"\bseventeen\b": "17", r"\beighteen\b": "18",
    r"\bnineteen\b": "19",

    # English tens
    r"\btwenty\b": "20", r"\bthirty\b": "30", r"\bforty\b": "40", r"\bfifty\b": "50",
    r"\bsixty\b": "60", r"\bseventy\b": "70", r"\beighty\b": "80", r"\bninety\b": "90",

    # English larger units
    r"\bhundred\b": "100", r"\bthousand\b": "1000", r"\bmillion\b": "1000000", r"\bbillion\b": "1000000000",

    # English ops/keywords
    r"\bplus\b": "+", r"\bminus\b": "-", r"\btimes\b": "*", r"\bmultiply\b": "*",
    r"\bmultiplied by\b": "*", r"\bdivide\b": "/", r"\bdivided by\b": "/", r"\bover\b": "/",
    r"\bequals?\b": "=", r"\bpercent(age)?\b": "%", r"\bsquare root\b": "sqrt", r"\bsqrt\b": "sqrt",
    r"\bpoint\b": ".", r"\bcomma\b": ".", r"\bpower\b": "^", r"\bto the power of\b": "^",
    r"\bsquared\b": "^2", r"\bcubed\b": "^3",
    r"\bof\b": "*",  # "percent of" or general multiplication

    # Arabic digits
    r"\bصفر\b": "0", r"\bواحد\b": "1",
    r"\bاثنان\b": "2", r"\bاثنين\b": "2",
    r"\bثلاثة\b": "3", r"\bثلاث\b": "3",
    r"\bأربعة\b": "4", r"\bاربعة\b": "4", r"\bاربع\b": "4",
    r"\bخمسة\b": "5", r"\bخمس\b": "5",
    r"\bستة\b": "6", r"\bست\b": "6",
    r"\bسبعة\b": "7", r"\bسبع\b": "7",
    r"\bثمانية\b": "8", r"\bثماني\b": "8",
    r"\bتسعة\b": "9", r"\bتسع\b": "9",
    r"\bعشرة\b": "10", r"\bعشر\b": "10",

    # Arabic teens
    r"\bأحد عشر\b": "11", r"\bاحد عشر\b": "11",
    r"\bإثنا عشر\b": "12", r"\bاثنا عشر\b": "12", r"\bإثني عشر\b": "12", r"\bاثني عشر\b": "12",
    r"\bثلاثة عشر\b": "13", r"\bثلاثه عشر\b": "13",
    r"\bأربعة عشر\b": "14", r"\bاربعة عشر\b": "14",
    r"\bخمسة عشر\b": "15", r"\bخمسه عشر\b": "15",
    r"\bستة عشر\b": "16", r"\bسته عشر\b": "16",
    r"\bسبعة عشر\b": "17", r"\bسبعه عشر\b": "17",
    r"\bثمانية عشر\b": "18", r"\bثمانيه عشر\b": "18",
    r"\bتسعة عشر\b": "19", r"\bتسعه عشر\b": "19",

    # Arabic tens
    r"\bعشرون\b": "20", r"\bعشرين\b": "20",
    r"\bثلاثون\b": "30", r"\bثلاثين\b": "30",
    r"\bأربعون\b": "40", r"\b اربعون\b": "40", r"\bأربعين\b": "40", r"\bاربعين\b": "40",
    r"\bخمسون\b": "50", r"\bخمسين\b": "50",
    r"\bستون\b": "60", r"\bستين\b": "60",
    r"\bسبعون\b": "70", r"\bسبعين\b": "70",
    r"\bثمانون\b": "80", r"\bثمانين\b": "80",
    r"\bتسعون\b": "90", r"\bتسعين\b": "90",

    # Arabic larger units
    r"\bمئة\b": "100", r"\bمائة\b": "100", r"\bميه\b": "100",
    r"\bألف\b": "1000", r"\bالف\b": "1000",
    r"\bمليون\b": "1000000",
    r"\bمليار\b": "1000000000",

    # Arabic ops
    r"\bزائد\b": "+", r"\bناقص\b": "-", r"\bضرب\b": "*",
    r"\bقسمة\b": "/", r"\bعلى\b": "/", r"\bيساوي\b": "=",
    r"\bنسبة\b": "%", r"\bبالمئة\b": "%", r"\bفي المئة\b": "%",
    r"\bجذر\b": "sqrt", r"\bتربيعي\b": "sqrt",
    r"\bنقطة\b": ".", r"\bفاصلة\b": ".",
    r"\bأس\b": "^", r"\bقوة\b": "^",
    r"\bمن\b": "*",  # often spoken like "50 بالمئة من 200"
}


def normalize_text_numbers_ops(text: str) -> str:
    s = text
    # Lowercase English letters only
    s = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), s)
    # Strip Arabic definite article "ال" before certain math words
    s = re.sub(
        r"\bال(?=(جذر|نسبة|فاصلة|نقطة|قسم|قسمة|ضرب|جمع|طرح|تربيعي))",
        "",
        s)
    # Apply bilingual word map
    for pat, repl in WORD_MAP.items():
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    # Normalize spacing
    s = re.sub(r"\s+", " ", s).strip()
    return s


def postprocess_to_expression(s: str) -> str:
    # Convert ^ to ** for Python/SymPy power
    s = re.sub(r"\^", "**", s)

    # Convert percentages: 50% -> (50/100)
    s = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1/100)", s)

    # Make sure 'sqrt 9' -> 'sqrt(9)'
    s = re.sub(r"\bsqrt\s*\(\s*", "sqrt(", s)     # already has '('
    s = re.sub(r"\bsqrt\s+([0-9.\-+*/()]+)", r"sqrt(\1)", s)

    # Remove equals side if present: keep RHS evaluation or LHS?
    # If there's '=', prefer evaluating the side after '='; else before '='.
    if "=" in s:
        parts = s.split("=")
        # choose RHS if non-empty else LHS
        s = parts[-1].strip() if parts[-1].strip() else parts[0].strip()

    # Remove any characters not allowed (digits, ops, dot, parens, letters in
    # sqrt)
    s = re.sub(r"[^0-9+\-*/().sqrtt ]", "", s)

    # Condense 's q r t' to 'sqrt'
    s = re.sub(r"s\s*q\s*r\s*t", "sqrt", s, flags=re.I)

    # Balance parentheses lightly: add missing closing
    opens = s.count("(")
    closes = s.count(")")
    if opens > closes:
        s += ")" * (opens - closes)

    # Collapse spaces around operators
    s = re.sub(r"\s*([+\-*/()])\s*", r"\1", s)
    return s.strip()


# ====== Safe evaluation with SymPy ======
ALLOWED_FUNCS = {"sqrt": sqrt}


def evaluate_expression(expr: str):
    # Reject if unknown letters appear (only allow 'sqrt')
    letters = re.findall(r"[A-Za-z]+", expr)
    if any(tok.lower() not in ALLOWED_FUNCS for tok in letters):
        raise ValueError(f"Unsupported token(s): {letters}")
    try:
        tree = sympify(expr, locals=ALLOWED_FUNCS, evaluate=True)
    except SympifyError as e:
        raise ValueError(f"Cannot parse expression: {expr}") from e
    # Numerical evaluation (high precision)
    val = N(tree, 12)
    return val

# ====== Audio helpers ======


def record_audio():
    key = f"audio_rec_{st.session_state.get('audio_key', 0)}"
    audio_value = st.audio_input("Record a voice message", key=key)

    if audio_value is not None:
        data, sr = sf.read(BytesIO(audio_value.getvalue()))

        if data.dtype != np.float32:
            data = data.astype(np.float32)

        audio_for_whisper = librosa.resample(
            data,
            orig_sr=sr,
            target_sr=TARGET_SAMPLE_RATE
        )
        return audio_for_whisper

    return None

# ====== Transcription ======


def transcribe(audio_for_whisper, lang, initial_prompt):

    forced_decoder_ids = st.session_state["proc"].get_decoder_prompt_ids(
        language=lang, task="transcribe")

    #forced_decoder_ids = st.session_state["proc"].get_decoder_prompt_ids(
    #    language=lang, task="translate")
    
    inputs = st.session_state["proc"](
        audio_for_whisper,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        prompt_ids=initial_prompt)
    
    input_model = inputs.input_features

    predicted_ids = st.session_state["model"].generate(
        input_model,
        forced_decoder_ids=forced_decoder_ids,
        num_beams=3,
        #no_repeat_ngram_size=2,
        early_stopping=True
    )

    
    transcription = st.session_state["proc"].batch_decode(
        predicted_ids)  # decode token ids to text
    

    raw = transcription[0]
    clean = normalize_text_numbers_ops(raw)
    expr = postprocess_to_expression(clean)

    return raw, clean, expr

# ====== Main Function ======

def main():
    st.set_page_config(
        page_title="BISCA",
        layout="wide"
    )

    # Display project name title at the top
    st.markdown("<h1 style='text-align: center; margin-bottom: 25px;'>BISCA</h1>", unsafe_allow_html=True)

    # Left-side touch buttons (Arabic, English, Start, Restart)
    with st.sidebar:
        st.markdown("<h2>Menu</h2>", unsafe_allow_html=True)
        st.markdown("---")
        lang_choice = st.radio("Language", ["Arabic", "English"], horizontal=False)
        st.markdown("---")
        start_pressed = st.button("Start")
        restart_pressed = st.button("Restart")

    # Session state setup
    proc, model = load_models()

    if proc not in st.session_state:
        st.session_state["proc"] = proc

    if model not in st.session_state:
        st.session_state["model"] = model

    if "initial_prompt" not in st.session_state:
        st.session_state["initial_prompt"] = []

    if "started" not in st.session_state:
        st.session_state["started"] = False

    if "last_lang" not in st.session_state:
        st.session_state["last_lang"] = None

    if "audio" not in st.session_state:
        st.session_state["audio"] = None

    # Handle language change
    if lang_choice != st.session_state.get("last_lang"):
        st.session_state["started"] = False
        st.session_state["initial_prompt"] = []
        st.session_state["last_lang"] = lang_choice


    # Start button logic
    if start_pressed:
        st.session_state["started"] = True
        st.session_state["audio"] = None

        if lang_choice == "Arabic":
            st.session_state["initial_prompt"] = INITIAL_PROMPT_ar
        else:
            st.session_state["initial_prompt"] = INITIAL_PROMPT_en

    # Restart button logic
    if restart_pressed:
        st.session_state["initial_prompt"] = []
        st.session_state["audio"] = None
        st.session_state["started"] = True
        st.session_state["audio_key"] = (st.session_state.get("audio_key") or 0) + 1

    st.markdown("---")


    with st.expander("About the BISCA", expanded=True):
        st.write(
            """
            - The UI of the BISCA was built using Streamlit.
            - ASR (Automatic Speech Recognition) was implemented using OpenAI's Whisper Fine-Tuned for our needs.
            - Calculations are generated using our own functions.
            """
        )

    if st.session_state.get("started"):
   
        if st.session_state["audio"] is None:
            st.session_state["audio"] = record_audio()

        audio = st.session_state["audio"]

        if audio is not None:
            with st.spinner("Transcribing ...", show_time=True):
                raw, clean, expr = transcribe(audio, lang_choice, st.session_state["initial_prompt"])

            try:
                result = evaluate_expression(expr)
                print(f'Raw transcription: {raw}')
                print(f'Normalized text: {clean}')
                print(f'Expression: {expr}')

                st.markdown(
                    f"""
                    <h2 style="text-align:center; font-size:50px;">
                        The result is <span style="color:#90ee90;">{result:.2f}</span>
                    </h2>
                    """,
                    unsafe_allow_html=True
                )



            except Exception as e:
                st.error(f'{e}')
                st.error(f'Raw text: {raw}')
        else:
            st.info("Please record your voice to start the calculation.")
    else:
        st.warning("Click 'Start' to begin the voice calculator.")


if __name__ == "__main__":
    main()

