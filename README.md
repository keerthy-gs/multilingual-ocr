# Zero-Shot Multilingual Scene Text Detection for Indian Languages

An end-to-end AI pipeline that detects, recognises, translates, and vocalises text from Indian signboard images across multiple scripts.

## Features
- Zero-shot script identification using CLIP (ViT-B/32)
- Supports 5 scripts: Kannada, Telugu, Hindi, Bengali, English
- Multiple EasyOCR language readers process the image, and the result with the highest OCR confidence is selected
- Neural machine translation via Google Translate API
- Text-to-speech voice output using gTTS
- Interactive web app built with Streamlit

## Tech Stack
- Python 3.14
- CLIP (ViT-B/32)
- EasyOCR
- OpenCV
- Streamlit
- Google Translate API
- gTTS

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/keerthy-gs/multilingual-ocr.git
cd multilingual-ocr
```

### 2. Install the Required Packages

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Application

```bash
streamlit run app.py
```

The first launch may take longer while the OCR and language models load.
## Example Outputs

- Achieved a 92.4% OCR confidence score on one tested Kannada signboard image.
- Detected six text regions in one code-mixed Bengali-English sample image.
- Demonstrated cross-language translation, including Bengali-to-Kannada translation.

> Note: These are demonstration results from selected sample images and do not represent overall benchmark accuracy.
