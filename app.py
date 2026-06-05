import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2
import tempfile

from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator
from gtts import gTTS


# -------------------------------------------------
# LOAD CLIP MODEL
# -------------------------------------------------

@st.cache_resource
def load_clip():

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    return clip_model, clip_processor


clip_model, clip_processor = load_clip()

# -------------------------------------------------
# LOAD OCR READERS
# -------------------------------------------------

@st.cache_resource
def load_readers():

    reader_en = easyocr.Reader(['en'])

    reader_te = easyocr.Reader(['te', 'en'])

    reader_hi = easyocr.Reader(['hi', 'en'])

    reader_kn = easyocr.Reader(['kn', 'en'])

    reader_bn = easyocr.Reader(['bn', 'en'])

    detection_reader = easyocr.Reader(['en'])

    return (
        reader_en,
        reader_te,
        reader_hi,
        reader_kn,
        reader_bn,
        detection_reader
    )


(
    reader_en,
    reader_te,
    reader_hi,
    reader_kn,
    reader_bn,
    detection_reader
) = load_readers()

# -------------------------------------------------
# UNICODE HELPERS
# -------------------------------------------------

def count_telugu(text):

    count = 0

    for ch in text:

        if '\u0C00' <= ch <= '\u0C7F':

            count += 1

    return count


def count_kannada(text):

    count = 0

    for ch in text:

        if '\u0C80' <= ch <= '\u0CFF':

            count += 1

    return count


def count_hindi(text):

    count = 0

    for ch in text:

        if '\u0900' <= ch <= '\u097F':

            count += 1

    return count


def detect_script_from_text(text):

    hindi_chars = sum(
        1 for c in text
        if '\u0900' <= c <= '\u097F'
    )

    telugu_chars = sum(
        1 for c in text
        if '\u0C00' <= c <= '\u0C7F'
    )

    kannada_chars = sum(
        1 for c in text
        if '\u0C80' <= c <= '\u0CFF'
    )

    tamil_chars = sum(
        1 for c in text
        if '\u0B80' <= c <= '\u0BFF'
    )

    english_chars = sum(
        1 for c in text
        if c.isascii() and c.isalpha()
    )

    scores = {

        "Hindi": hindi_chars,

        "Telugu": telugu_chars,

        "Kannada": kannada_chars,

        "Tamil": tamil_chars,

        "English": english_chars

        
    }

    return max(
        scores,
        key=scores.get
    )

# -------------------------------------------------
# CLEAN TEXT
# -------------------------------------------------

def clean_text(text):

    text = text.replace("\n", " ")

    text = text.strip()

    return text


def is_garbage_text(text):

    text = text.strip()

    if len(text) <= 2:
        return True

    if text.isdigit():
        return True

    symbol_count = sum(
        not ch.isalnum() and not ch.isspace()
        for ch in text
    )

    if len(text) > 0 and symbol_count / len(text) > 0.5:
        return True

    return False
# -------------------------------------------------
# BOARD CLASSIFIER
# -------------------------------------------------

def classify_board(text):

    text = text.upper()

    prohibition_words = [
        "NO ENTRY",
        "PROHIBITED",
        "NOT ALLOWED",
        "RESTRICTED",
        "WITHOUT PERMISSION"
    ]

    warning_words = [
        "WARNING",
        "DANGER",
        "CAUTION",
        "HIGH VOLTAGE",
        "RISK"
    ]

    direction_words = [
        "LEFT",
        "RIGHT",
        "EXIT",
        "WAY",
        "ROAD",
        "THIS WAY"
    ]

    government_words = [
        "GOVERNMENT",
        "MUNICIPAL",
        "OFFICE",
        "DEPARTMENT",
        "CORPORATION"
    ]

    commercial_words = [
        "HOTEL",
        "RESTAURANT",
        "SHOP",
        "STORE",
        "MALL"
    ]

    for word in prohibition_words:

        if word in text:

            return "Prohibition Board"

    for word in warning_words:

        if word in text:

            return "Warning Board"

    for word in direction_words:

        if word in text:

            return "Direction Board"

    for word in government_words:

        if word in text:

            return "Government Board"

    for word in commercial_words:

        if word in text:

            return "Commercial Board"

    return "Information Board"

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.title(
    "Indian Multilingual Scene Text System"
)

st.write(
    "Upload an image containing multilingual scene text."
)
language_codes = {

            "English":"en",
            "Hindi":"hi",
            "Telugu":"te",
            "Kannada":"kn",
            "Tamil":"ta",
            "Malayalam":"ml",
            "Bengali":"bn",
            "Gujarati":"gu",
            "Punjabi":"pa"
}




show_uploaded_image = st.checkbox(
    "Show Uploaded Image"
)

show_bounding_boxes = st.checkbox(
    "Show Bounding Boxes"
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# MAIN
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    if show_uploaded_image:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # -------------------------------------------------
    # DETECT TEXT REGIONS
    # -------------------------------------------------

    detection_results = detection_reader.readtext(
        image_np,
        paragraph=False
    )

    boxed_image = image_bgr.copy()

    final_outputs = []

    # -------------------------------------------------
    # PROCESS REGIONS
    # -------------------------------------------------

    for result in detection_results:

        bbox = result[0]

        x1 = int(min([p[0] for p in bbox]))
        y1 = int(min([p[1] for p in bbox]))

        x2 = int(max([p[0] for p in bbox]))
        y2 = int(max([p[1] for p in bbox]))

        padding = 12

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)

        x2 = min(image_np.shape[1], x2 + padding)
        y2 = min(image_np.shape[0], y2 + padding)

        width = x2 - x1
        height = y2 - y1

        # -------------------------------------------------
        # REMOVE VERY SMALL REGIONS
        # -------------------------------------------------

        if width < 25 or height < 12:
            continue

        crop = image_np[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop)

        # -------------------------------------------------
        # CLIP ZERO SHOT
        # -------------------------------------------------

        candidate_labels = [

            "Telugu script",

            "Kannada script",

            "Hindi script",

            "English script"
        ]

        inputs = clip_processor(

            text=candidate_labels,

            images=crop_pil,

            return_tensors="pt",

            padding=True
        )

        outputs = clip_model(**inputs)

        probs = outputs.logits_per_image.softmax(
            dim=1
        )

        clip_prediction = candidate_labels[
            probs.argmax()
        ]

        # -------------------------------------------------
        # OCR COMPETITION
        # -------------------------------------------------

        readers = {

            "English": reader_en,

            "Telugu": reader_te,

            "Hindi": reader_hi,

            "Kannada": reader_kn,

            "Bengali": reader_bn
        }

        best_text = ""

        best_language = ""

        best_score = 0

        winner_confidence = 0

        for lang_name, reader in readers.items():

            try:

                region_results = reader.readtext(
                    crop,
                    detail=1,
                    paragraph=False
                )

                if len(region_results) == 0:
                    continue

                combined_text = ""

                total_conf = 0

                count = 0

                for r in region_results:

                    combined_text += r[1] + " "

                    total_conf += r[2]

                    count += 1

                avg_conf = total_conf / count

                
                final_score = (
                    avg_conf 
                    
                )

                combined_text = clean_text(
                    combined_text
                )
                
                digit_ratio = (
                    sum(c.isdigit() for c in combined_text)
                    / max(len(combined_text), 1)
                )
                if digit_ratio > 0.40:
                    continue
                
                if final_score > best_score:

                    best_score = final_score
                  
                    winner_confidence = avg_conf

                    best_language = lang_name

                    best_text = combined_text

            except:

                pass

        # -------------------------------------------------
        # REMOVE BAD DETECTIONS
        # -------------------------------------------------

        if best_score < 0.25:
            continue

        if is_garbage_text(best_text):
            continue

        # -------------------------------------------------
        # ENGLISH SCRIPT VALIDATION
        # -------------------------------------------------

        english_chars = sum(
            c.isascii() and c.isalpha()
            for c in best_text
        )

        english_ratio = english_chars / max(
            len(best_text),
            1
        )

        if (
            len(best_text) >= 4
            and english_ratio > 0.85
        ):
            best_language = "English"

        # -------------------------------------------------
        # STORE OUTPUT
        # -------------------------------------------------

        final_outputs.append({

            "language": best_language,

            "clip_prediction": clip_prediction,

            "confidence": round(
                winner_confidence * 100,
                2
            ),

            "text": best_text,

            "bbox": [x1, y1, x2, y2]
        })



    # -------------------------------------------------
    # DISPLAY OUTPUTS
    # -------------------------------------------------
    final_outputs = sorted(
        final_outputs,
        key=lambda x: (
             x["bbox"][1],
                x["bbox"][0]
        )
    )
          

    st.subheader(
        "Detected Multilingual Outputs"
    )

    for idx, item in enumerate(final_outputs):

        x1, y1, x2, y2 = item["bbox"]

        # -------------------------------------------------
        # DRAW BOXES
        # -------------------------------------------------

        if show_bounding_boxes:

            cv2.rectangle(

                boxed_image,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )

            cv2.putText(

                boxed_image,

                item["language"],

                (x1, y1 - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 0, 0),

                2
            )

        st.write(
            f"[{idx}] {item['language']}"
            
        )

        st.write(
            item["text"]
        )
        st.caption(
             f"CLIP Prediction: {item['clip_prediction']}"
        )
        st.caption(
             f"OCR Winner: {item['language']}"
        )
        st.caption(
            f"Confidence: {item['confidence']}%"
        )
        st.write("---")

    # -------------------------------------------------
    # SHOW BOX IMAGE
    # -------------------------------------------------

    if show_bounding_boxes:

        boxed_rgb = cv2.cvtColor(
            boxed_image,
            cv2.COLOR_BGR2RGB
        )

        st.image(

            boxed_rgb,

            caption="Detected Regions",

            use_container_width=True
        )
        # -------------------------------------------------
# USER TEXT SELECTION
# -------------------------------------------------

    if len(final_outputs) > 0:

        available_languages = sorted(
            list(
                set(
                    item["language"]
                    for item in final_outputs
                )
            )
        )

        selected_language = st.selectbox(
            "Select Script / Language",
            available_languages
        )

        st.success(
            f"Detected Language: {selected_language}"
        )

        language_texts = []

        for item in final_outputs:

            if item["language"] == selected_language:

                language_texts.append(
                    item["text"]
                )

        selected_text = "\n".join(
            language_texts
        )

        board_type = classify_board(

            selected_text
         )
        st.success(
            f"Board Type: {board_type}"
         )

        st.subheader(
            "Edit Detected Text"
        )

        edited_text = st.text_area(
            "Modify text if needed",
            value=selected_text,
            height=200
        )

        confirm_text = st.checkbox(
            "I confirm this text is correct"
        )

        if confirm_text:

            target_language = st.selectbox(
                "Translate To",
                list(language_codes.keys())
            )

            translated_text = GoogleTranslator(
                source='auto',
                target=language_codes[target_language]
            ).translate(
                edited_text
            )

            st.subheader(
                "Translated Output"
            )

            st.write(
                translated_text
            )

            meaningful_sentence = st.text_area(
                "Edit final translated text before voice",
                value=translated_text
            )

            if st.button(
                "Generate Voice"
            ):

                tts = gTTS(
                    text=meaningful_sentence,
                    lang=language_codes[target_language]
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3"
                ) as tmp_audio:

                    tts.save(
                        tmp_audio.name
                    )

                    audio_file = tmp_audio.name

                st.audio(
                    audio_file
                )
       