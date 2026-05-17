from langchain_core.documents import Document
from typing import List
import pdfplumber


def load_from_pdf(pdf_path: str,
                  split_from_mid_page: List[int],
                  **kwargs) -> List[Document]:
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page in split_from_mid_page:
                left_column = page.crop((0, 0, 0.5 * page.width, page.height))
                right_column = page.crop((0.5 * page.width, 0, page.width, page.height))
                text = (left_column.extract_text(**kwargs) +
                        right_column.extract_text(**kwargs)).replace("\n"," ")
                docs.append(Document(page_content=text))
            else:
                docs.append(Document(page_content=page.extract_text(**kwargs).replace("\n"," ")))
    for page_num, doc in enumerate(docs):
        doc.metadata["source"] = page_num+1
    return docs

if __name__ == "__main__":
    import nltk

    pdf_path = "MRM_toolkit/data/wp_generative_ai_risk_management_in_fs.pdf"
    page_split_from_mid = list(range(4,13))
    doc = load_from_pdf(pdf_path=pdf_path, split_from_mid_page=page_split_from_mid)
    sentences = nltk.sent_tokenize(doc[3].page_content)
    print(sentences)


