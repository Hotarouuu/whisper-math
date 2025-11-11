import queue, sys, re, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel
from sympy import sympify, sqrt, N
from sympy.core.sympify import SympifyError
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

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

# ====== Audio helpers ======
def print_devices():
    print(sd.query_devices())

def record_until_enter():
    print("\nPress ENTER to START, then ENTER again to STOP…")
    input()
    print("Recording… (press ENTER to stop)")
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="float32", callback=cb, device=DEVICE)
    stream.start()
    try:
        input()
    finally:
        stream.stop(); stream.close()

    chunks = []
    while not q.empty():
        chunks.append(q.get())
    return np.concatenate(chunks, axis=0).flatten() if chunks else np.zeros((0,), dtype=np.float32)

# ====== Text normalization (EN + AR → mathy string) ======
WORD_MAP = {
    # English digits
    r"\bzero\b": "0", r"\bone\b": "1", r"\btwo\b": "2", r"\bthree\b": "3", r"\bfour\b": "4",
    r"\bfive\b": "5", r"\bsix\b": "6", r"\bseven\b": "7", r"\beight\b": "8", r"\bnine\b": "9",
    r"\bten\b": "10",
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
    r"\bثلاثة\b": "3",
    r"\bأربعة\b": "4", r"\bاربعة\b": "4",
    r"\bخمسة\b": "5",
    r"\bستة\b": "6",
    r"\bسبعة\b": "7",
    r"\bثمانية\b": "8",
    r"\bتسعة\b": "9",
    r"\bعشرة\b": "10",
    # Arabic ops
    r"\bزائد\b": "+", r"\bناقص\b": "-", r"\bضرب\b": "*",
    r"\bقسمة\b": "/", r"\bعلى\b": "/", r"\bيساوي\b": "=",
    r"\bنسبة\b": "%", r"\bمئة\b": "100",
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
    s = re.sub(r"\bال(?=(جذر|نسبة|فاصلة|نقطة|قسم|قسمة|ضرب|جمع|طرح|تربيعي))", "", s)
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

    # Remove any characters not allowed (digits, ops, dot, parens, letters in sqrt)
    s = re.sub(r"[^0-9+\-*/().sqrtt ]", "", s)

    # Condense 's q r t' to 'sqrt'
    s = re.sub(r"s\s*q\s*r\s*t", "sqrt", s, flags=re.I)

    # Balance parentheses lightly: add missing closing
    opens = s.count("("); closes = s.count(")")
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

# ====== Main loop ======
def main():

    # Temporary language selection

    lang = input(str("Select language (en or ar): ")).lower()
    if lang in ['ar', 'arabic', 'english', 'en']:
        print(f'Language Selected: {lang} ')

    print("Loading ASR model…")

    processor = AutoProcessor.from_pretrained("manushya-ai/whisper-medium-finetuned")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("manushya-ai/whisper-medium-finetuned")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")

    if lang == 'ar' or lang == 'arabic':
        prompt_ids = processor.tokenizer.get_prompt_ids(
        INITIAL_PROMPT_ar, 
        )
    else:
        prompt_ids = processor.tokenizer.get_prompt_ids(
        INITIAL_PROMPT_en, 
        )        

    while True:
        audio = record_until_enter()
        if audio.size == 0:
            print("No audio captured. Try again.")
            continue


        audio_for_whisper = librosa.resample(
        audio, 
        orig_sr=SAMPLE_RATE, 
        target_sr=TARGET_SAMPLE_RATE
        )

        print("Transcribing…")

        input_features = processor(audio_for_whisper, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", prompt_ids=prompt_ids).input_features
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids) # generate token ids
        transcription = processor.batch_decode(predicted_ids) # decode token ids to text

        raw = transcription[0]
        clean = normalize_text_numbers_ops(raw)
        expr = postprocess_to_expression(clean)

        print("\n--- TRANSCRIPT ---")
        print("Raw   :", raw)
        print("Clean :", clean)
        print("Expr  :", expr)

        try:
            result = evaluate_expression(expr)
            print("Result:", result)
        except Exception as e:
            print("Error :", e)

        print("------\nPress Ctrl+C to exit, or do another recording.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")

# note: the function is not parsing complex numbers like 125. Only basic numbers like 1000 or 100