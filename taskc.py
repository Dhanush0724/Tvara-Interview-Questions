# task_c_vectorization.py
# Task C — Vectorization with Hugging Face (intfloat/e5-small-v2)

from sentence_transformers import SentenceTransformer, util
import torch

def main():
    # 1. Load the tiny embedding model from Hugging Face
    model = SentenceTransformer("intfloat/e5-small-v2")

    # 2. Example sentences (can be replaced with PDF text later)
    sentences = [
        "Artificial intelligence is transforming modern software systems.",
        "Machine learning models require large amounts of quality data.",
        "Natural language processing enables computers to understand text.",
        "Cloud computing allows scalable deployment of applications.",
        "Cybersecurity is critical for protecting sensitive information."
    ]

    # E5 requires prefixing
    passages = [f"passage: {s}" for s in sentences]

    # 3. Generate embeddings for passages
    passage_embeddings = model.encode(
        passages,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # 4. Query for similarity search
    query = "which is criticial for sesitive information?"
    query_embedding = model.encode(
        f"query: {query}",
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # 5. Compute cosine similarity
    scores = util.cos_sim(query_embedding, passage_embeddings)[0]

    # 6. Get most relevant sentence
    best_idx = torch.argmax(scores).item()

    print("Query:")
    print(query)
    print("\nMost relevant sentence:")
    print(sentences[best_idx])
    print("\nSimilarity score:")
    print(float(scores[best_idx]))


if __name__ == "__main__":
    main()


# task_c_vectorization.py
# Task C — Vectorization with Hugging Face (intfloat/e5-small-v2)
# Includes PDF text extraction

# from sentence_transformers import SentenceTransformer, util
# import torch
# from PyPDF2 import PdfReader


# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file"""
#     reader = PdfReader(pdf_path)
#     text = []

#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             # Split into sentences/lines
#             lines = page_text.split("\n")
#             text.extend([line.strip() for line in lines if line.strip()])

#     return text


# # def main():
# #     # 1. Load embedding model
# #     model = SentenceTransformer("intfloat/e5-small-v2")

# #     # 2. Extract text from sample PDF
# #     pdf_path = "sample.pdf"   # <-- place your PDF in the same folder
# #     sentences = extract_text_from_pdf(pdf_path)

# #     # Keep only first 3–5 lines as required
# #     sentences = sentences[:5]

# #     if not sentences:
# #         print("No text extracted from PDF.")
# #         return

# #     # E5 requires passage prefix
# #     passages = [f"passage: {s}" for s in sentences]

# #     # 3. Generate embeddings
# #     passage_embeddings = model.encode(
# #         passages,
# #         convert_to_tensor=True,
# #         normalize_embeddings=True
# #     )

# #     # 4. Query
# #     query = "How do computers understand human language?"
# #     query_embedding = model.encode(
# #         f"query: {query}",
# #         convert_to_tensor=True,
# #         normalize_embeddings=True
# #     )

# #     # 5. Similarity search
# #     scores = util.cos_sim(query_embedding, passage_embeddings)[0]
# #     best_idx = torch.argmax(scores).item()

# #     # 6. Output
# #     print("Query:")
# #     print(query)

# #     print("\nExtracted PDF Sentences:")
# #     for i, s in enumerate(sentences):
# #         print(f"{i+1}. {s}")

# #     print("\nMost Relevant Sentence:")
# #     print(sentences[best_idx])

# #     print("\nSimilarity Score:")
# #     print(float(scores[best_idx]))


# # if __name__ == "__main__":
# #     main()
