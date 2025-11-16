from pathlib import Path
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer
)

# Configure base model and save directory for compressed model
model_id = "manushya-ai/whisper-medium-finetuned"
save_dir = "C:\\Users\\lucar-work\\Documents\\GitHub\\whisper-math\\whisper-quantized"

# Export model in ONNX
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
model_dir = model.model_save_dir

# Run quantization for all ONNX files of exported model
onnx_models = list(Path(model_dir).glob("*.onnx"))
print(onnx_models)
quantizers = [ORTQuantizer.from_pretrained(model_dir, file_name=onnx_model) for onnx_model in onnx_models]

qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

for quantizer in quantizers:
    # Apply dynamic quantization and save the resulting model
    quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

# The source code is from https://medium.com/@daniel-klitzke/quantizing-openais-whisper-with-the-huggingface-optimum-library-30-faster-inference-64-36d9815190e0

