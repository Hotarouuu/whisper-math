from datetime import datetime
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import GenerationConfig, pipeline, AutoProcessor
import torchaudio
import librosa

# Number of inferences for comparing timings
num_inferences = 4
save_dir = "C:\\Users\\lucar-work\\Documents\\GitHub\\whisper-math\\whisper-quantized"
inference_file = "C:\\Users\\lucar-work\\Documents\\GitHub\\whisper-math\\data\\processed data\\arabic\\B\\arabic 1.wav"

def transcribe(model, audio_for_whisper, lang, initial_prompt):

    audio_for_whisper = torchaudio.load(audio_for_whisper)

    audio_for_whisper = audio_for_whisper[0].numpy()

    audio_for_whisper = librosa.resample(
    audio_for_whisper,
    orig_sr=48000,
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
    providers=["DmlExecutionProvider"],  
    encoder_file_name="encoder_model_quantized.onnx",
    decoder_file_name="decoder_model_quantized.onnx",
    decoder_with_past_file_name="decoder_with_past_model_quantized.onnx"
)

model.to("cpu")
proc = AutoProcessor.from_pretrained(save_dir)
# Create pipeline with original model as baseline
cls_pipeline_original = pipeline("automatic-speech-recognition", model="manushya-ai/whisper-medium-finetuned", device="cpu")

# Measure inference of quantized model
start_quantized = datetime.now()
for i in range(num_inferences):
    quantized = transcribe(model, inference_file, lang="ar", initial_prompt=None)
end_quantized = datetime.now()

# Measure inference of original model
start_original = datetime.now()
for i in range(num_inferences):
    not_quantized = transcribe(cls_pipeline_original.model, inference_file, lang="ar", initial_prompt=None)
end_original = datetime.now()

original_inference_time = (end_original - start_original).total_seconds() / num_inferences
print(f"Original inference time: {original_inference_time}")
print(f"Original transcription: {not_quantized}")

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")
print(f"Quantized transcription: {quantized}")
