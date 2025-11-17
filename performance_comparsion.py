from datetime import datetime
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import GenerationConfig, pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import soundfile as sf
import librosa

# Number of inferences for comparing timings
num_inferences = 4
save_dir = "/workspace/whisper-math/new_whisper_quantized"
inference_file = "/workspace/whisper-math/data/processed data/arabic/B/arabic 1.wav"

def transcribe(model, audio_for_whisper, lang, initial_prompt):

    audio, sr = sf.read(audio_for_whisper)

    audio_for_whisper = audio

    audio_for_whisper = librosa.resample(
    audio_for_whisper,
    orig_sr=sr,
    target_sr=16000
    )

    forced_decoder_ids = proc.get_decoder_prompt_ids(
        language=lang, task="transcribe")
    
    input_features = proc(
        audio_for_whisper,
        return_tensors="pt",
        prompt_ids=initial_prompt).input_features
    
    gen_config = GenerationConfig(forced_decoder_ids=forced_decoder_ids)

        
    predicted_ids = model.generate(
        input_features, generation_config=gen_config)  # generate token ids
    
    transcription = proc.batch_decode(
        predicted_ids)  # decode token ids to text

    raw = transcription[0]

    return raw

# Create pipeline based on quantized ONNX model
model = ORTModelForSpeechSeq2Seq.from_pretrained(
    save_dir,
    providers=["CPUExecutionProvider"],  
    encoder_file_name="encoder_model_quantized.onnx",
    decoder_file_name="decoder_model_quantized.onnx",
    decoder_with_past_file_name="decoder_with_past_model_quantized.onnx"
)

model.to("cpu")
proc = AutoProcessor.from_pretrained("openai/whisper-medium")
# Create pipeline with original model as baseline
# Load model directly

model_not_quantized = AutoModelForSpeechSeq2Seq.from_pretrained("manushya-ai/fiver-whisper-medium-finetuned")
# Measure inference of quantized model
start_quantized = datetime.now()
for i in range(num_inferences):
    quantized = transcribe(model, inference_file, lang="ar", initial_prompt=None)
end_quantized = datetime.now()

# Measure inference of original model
start_original = datetime.now()
for i in range(num_inferences):
    not_quantized = transcribe(model_not_quantized, inference_file, lang="ar", initial_prompt=None)
end_original = datetime.now()

original_inference_time = (end_original - start_original).total_seconds() / num_inferences
print(f"Original inference time: {original_inference_time}")
print(f"Original transcription: {not_quantized}")

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")
print(f"Quantized transcription: {quantized}")
