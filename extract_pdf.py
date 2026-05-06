import fitz
import sys

pdf_path = sys.argv[1]
doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text()

with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("PDF text extracted to pdf_text.txt")
