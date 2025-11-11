import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset, Audio,concatenate_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import pandas as pd
import warnings
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file

warnings.filterwarnings('ignore')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ====== Paths ======

data_path = os.getenv('data_path')

# ====== Importing the Model ======

model_name = "openai/whisper-medium"

processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(model_name)

# ====== Data Importing Function ======

def get_audio_file_paths(base_path_str: str) -> dict:
    
    base_path = Path(base_path_str)
    processed_dir = base_path / "processed data"
    
    audio_paths = {}
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}

    if not processed_dir.is_dir():
        return audio_paths

    for lang_dir in processed_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        
        lang_name = lang_dir.name
        audio_paths[lang_name] = {}
        
        for sub_dir in lang_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            
            sub_name = sub_dir.name
            
            files = [
                str(f.resolve()) for f in sub_dir.glob('*') 
                if f.is_file() and f.suffix.lower() in audio_extensions
            ]
            audio_paths[lang_name][sub_name] = files
            
    return audio_paths

# ====== Data Treatment Functions ======

def dataset_g(transcriptions, files, language : str):

    rows = []
    for label in files:
        text = transcriptions[label]  
        for file_path in files[label]:
            rows.append({'Label': label, 'audio': file_path, 'transcription': text, 'Language': language})

    df = pd.DataFrame(rows)

    df.drop('Label', axis=1, inplace=True)

    return df

def generate_audio_dataset(df_arabic, df_english, augment_factor=1):
  
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.005, p=0.3),
        TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
    ])

    dataset_arabic = Dataset.from_pandas(df_arabic).cast_column("audio", Audio(sampling_rate=16000))
    dataset_english = Dataset.from_pandas(df_english).cast_column("audio", Audio(sampling_rate=16000))

    def augment_audio(batch):
        audio_array = batch["audio"]["array"]
        if isinstance(audio_array, np.ndarray):
            augmented = augment(samples=audio_array, sample_rate=16000)
            batch["audio"] = {"array": augmented, "sampling_rate": 16000}
        return batch

    datasets_all = [dataset_arabic, dataset_english]

    for _ in range(augment_factor):
        ds_aug = dataset_arabic.map(augment_audio)
        datasets_all.append(ds_aug)

    full_dataset = concatenate_datasets(datasets_all)

    return full_dataset

def preprocess_function(batch):
    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=16000
        ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

# ====== Support Functions for Training ======

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
def compute_metrics(pred):
    metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():

    # Data Treatment
    
    text_a_english = "zero five twelve ninety-nine one hundred and five 2 plus 7 18 minus 4 6 times 3 20 divided by 5 ten plus thirty minus eight negative fifteen plus nine three to the power of two square root of sixteen clear equals repeat"
    text_a_arabic = "احسب خمسة زائد اثنين عشرة ناقص ثلاثة ستة ضرب أربعة عشرون قسمة خمسة سالب سبعة زائد واحد خمسة أس اثنين الجذر التربيعي لأربعة وعشرين امسح تأكيد أعِد calculate 37 plus خمسة اطرح twelve من عشرة اضرب ثلاثة في twenty eighty divided by ثمانية اجمع ١٢ و ١٣ سبعة زائد ١٩ 45 minus تسعة 3.5 plus اثنين ونصف واحد فاصلة خمسة ضرب أربعة مية واثنا عشر ناقص ستة 1000 minus 250 999 plus 1 قل اللون: أزرق"

    text_b_english = "one eight seventeen sixty-four one hundred and twenty 4 plus 9 22 minus 7 9 times 5 81 divided by 9 thirty plus fifty negative six minus ten plus three two to the power of five cube root of twenty-seven start stop undo"
    text_b_arabic = "اجمع سبعة و تلاتين مع 12 خمسة وأربعون ناقص عشرين تسعة ضرب ستة أربعة وستون قسمة ثمانية سالب ثلاثة زائد خمسة اثنان أس ثلاثة الجذر التكعيبي لسبعة وعشرين امسح الشاشة تم كرر آخر عملية calculate twelve times خمسة اقسم 36 على ستة اطرح خمسة من twenty fifty plus سبعة اجمع ١٠٠ و ٢٥ مئتان ناقص ٩٩ 14 minus أربعة اثنين فاصلة خمسة زائد 0.5 7.25 divided by خمسة أربع مية وخمسة ناقص عشرة 500 plus 500 1234 minus 234 قل اللون: أخضر"

    text_c_english = "two nine eleven seventy-three two hundred and three 8 plus 6 40 minus 12 7 times 7 90 divided by 10 twenty plus fifteen negative nine minus twenty plus eight five to the power of three square root of one hundred confirm repeat last slower please"
    text_c_arabic = "احسب 23 زائد 15 سبعة ناقص اثنين ثلاثة ضرب تسعة ستة وثلاثون قسمة أربعة سالب اثنا عشر زائد عشرة عشرة أس اثنين الجذر التربيعي لتسعة افتح رجوع أعد الحساب calculate twenty minus ثلاثة اجمع five و خمسة اضرب 8 في twenty-one thirty divided by ثلاثة اجمع ٧ و ١١ أربعون ناقص ١٨ 16 plus سبعة واحد فاصلة خمسة ناقص 0.25 2.2 times اثنين تسعمية وتسعة وتسعين زائد واحد 1500 minus 300 333 plus 667 قل اللون: أحمر"

    print(f'Importing data...')

    all_files = get_audio_file_paths(data_path)
    arabic_files = all_files['arabic']
    english_files = all_files['english']

    transcriptions_arabic = {
    'A': text_a_arabic,
    'B': text_b_arabic,
    'C': text_c_arabic
    }

    transcriptions_english = {
    'A': text_a_english,
    'B': text_b_english,
    'C': text_c_english
    }

    print('Treating data...')
    df_arabic = dataset_g(transcriptions_arabic, arabic_files, language='arabic')
    df_english = dataset_g(transcriptions_english, english_files, language='english')
    df_final = generate_audio_dataset(df_arabic, df_english, augment_factor=2)
    df_final = df_final.train_test_split(test_size=0.25)
    dataset = df_final.map(preprocess_function, remove_columns=df_final["train"].column_names)

    # Configuring parameters

    model.config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training

    print(f'Starting training...\n')

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-medium-finetuned",

        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5, 
        num_train_epochs=10, # best number of epochs!
        warmup_steps=30,

        gradient_checkpointing=False,
        bf16=True,
        fp16=False,

        evaluation_strategy="epoch",
        save_strategy="best",
        logging_strategy="steps",
        logging_steps=10,

        predict_with_generate=True,           
        generation_max_length=150,         

        dataloader_num_workers=0,             
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False
        report_to="tensorboard" # To save experiments and/or check the training logs
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")

