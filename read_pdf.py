import pypdf
import sys

def extract_text(pdf_path, output_path):
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(text)

if __name__ == "__main__":
    extract_text(sys.argv[1], sys.argv[2])
