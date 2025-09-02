# Manga Translator with Gemma 3n
## 🛠️ Tech Stacks
1. INPUT: Upload Manga Image (Streamlit) 
2. [Vision] Speech Bubble and Text Detection (YOLOv8)
3. [OCR] Text Recognition for detected Speech Bubble and text (PaddleOCR)
4. [NLP] Contextual Translation (Gemma 3n)
5. Text Length & Style Adjustments (Gemma 3n)
6. Replace the original text with the translated result (Pillow)
7. OUTPUT: Translated Manga Image (Streamlit)
## Credits
- Manga Speech Bubble and Text Detection Model - https://universe.roboflow.com/arjulaka/manga-text-and-bubble-detection-pizm9/
- Latin Text OCR - https://huggingface.co/PaddlePaddle/latin_PP-OCRv3_mobile_rec
- Japanese & Mandarin Text OCR - https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec
## (DISCONTINUED)
