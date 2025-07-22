# Importing required libraries
from PIL import Image
import fitz  # PyMuPDF
import os, json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Extracting images and text from pdf
def ext_content(pdf_path, output_folder="ext_images"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text().strip()
        image_paths = []

        for idx, image in enumerate(page.get_images(full=True)):
            xref = image[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = f"page{page_number}_img{idx + 1}.{img_ext}"
            img_path = os.path.join(output_folder, img_filename)

            with open(img_path, "wb") as f:
                f.write(img_bytes)

            image_paths.append(img_path)

        pages_data.append({
            "page_number": page_number,
            "text": page_text,
            "images": image_paths
        })

    with open("output.json", "w") as f:
        json.dump(pages_data, f, indent=2)

    return pages_data

# Using BLIP Model for generating questions from extracted content
def generate_questions(content):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to("cpu")

    all_images = []
    for page in content:
        all_images.extend(page["images"])

    questions = []

    group_size = 4
    for i in range(0, len(all_images), group_size):
        group = all_images[i:i + group_size]
        if len(group) < 4:
            break  

        question_img = group[0]
        options = group[1:]

        questions.append({
            "question": "What is the next figure?",
            "images": question_img,
            "option_images": options
        })

    # Step 3: Save to file
    with open("ai_generated_questions.json", "w") as f:
        json.dump(questions, f, indent=2)

    return questions

# Main function
if __name__ == "__main__":
    pdf_file = "IMO_Sample.pdf"
    extracted_data = ext_content(pdf_file)
    generate_questions(extracted_data)
