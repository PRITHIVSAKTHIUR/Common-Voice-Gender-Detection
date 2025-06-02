import gradio as gr
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa

# Load model and processor
model_name = "prithivMLmods/Common-Voice-Geneder-Detection"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "female",
    "1": "male"
}

def classify_audio(audio_path):
    # Load and resample audio to 16kHz
    speech, sample_rate = librosa.load(audio_path, sr=16000)

    # Process audio
    inputs = processor(
        speech,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio (WAV, MP3, etc.)"),
    outputs=gr.Label(num_top_classes=2, label="Gender Classification"),
    title="Common Voice Gender Detection",
    description="Upload an audio clip to classify the speaker's gender as female or male."
)

if __name__ == "__main__":
    iface.launch()
