![1.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/C-lq4SZvqsDgoppDfY4oh.png)

# Common-Voice-Gender-Detection

> **Common-Voice-Gender-Detection** is a fine-tuned version of `facebook/wav2vec2-base-960h` for **binary audio classification**, specifically trained to detect speaker gender as **female** or **male**. This model leverages the `Wav2Vec2ForSequenceClassification` architecture for efficient and accurate voice-based gender classification.

> [!note]
Wav2Vec2: Self-Supervised Learning for Speech Recognition : [https://arxiv.org/pdf/2006.11477](https://arxiv.org/pdf/2006.11477)

```py
Classification Report:

              precision    recall  f1-score   support

      female     0.9705    0.9916    0.9809      2622
        male     0.9943    0.9799    0.9870      3923

    accuracy                         0.9846      6545
   macro avg     0.9824    0.9857    0.9840      6545
weighted avg     0.9848    0.9846    0.9846      6545
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/pBnrVbG8uyZYq6Nb4GOuG.png)

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/QtZdfaXE-W-C4QDZUWuVC.png)

---

## Label Space: 2 Classes

```
Class 0: female  
Class 1: male
```

---

## Install Dependencies

```bash
pip install gradio transformers torch librosa hf_xet
```

---

## Inference Code

```python
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
```

---

## Demo Inference

> [!note]
male

<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/7woMf3_bgX_D99-1Uy3jH.mpga"></audio>

![Screenshot 2025-05-31 at 20-19-39 Common Voice Gender Detection.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/h1LqWmWbyi3ao2yvSWQSI.png)


> [!note]
female

<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/0d2rDf_DT-gjRWBwiPbm_.mpga"></audio>

![Screenshot 2025-05-31 at 20-21-57 Common Voice Gender Detection.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/TTAKrOOZ2sCS846wZWani.png)

--- 

## Intended Use

`Common-Voice-Gender-Detection` is designed for:

* **Speech Analytics** – Assist in analyzing speaker demographics in call centers or customer service recordings.
* **Conversational AI Personalization** – Adjust tone or dialogue based on gender detection for more personalized voice assistants.
* **Voice Dataset Curation** – Automatically tag or filter voice datasets by speaker gender for better dataset management.
* **Research Applications** – Enable linguistic and acoustic research involving gender-specific speech patterns.
* **Multimedia Content Tagging** – Automate metadata generation for gender identification in podcasts, interviews, or video content. 
