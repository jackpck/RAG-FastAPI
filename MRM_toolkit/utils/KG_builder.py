from ragas.testset.graph import Node
from ragas.testset.transforms.extractors import NERExtractor, KeyphrasesExtractor
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from typing import List
import asyncio
import nltk
nltk.download("punkt")

from MRM_toolkit.utils.loader import load_from_pdf

def build_nodes_from_pdf(pdf_path: str,
                         split_from_mid_page: List[int],
                         block_size: int = 4,
                         **kwargs) -> Node:
    docs = load_from_pdf(pdf_path=pdf_path, 
                         split_from_mid_page=split_from_mid_page, 
                         **kwargs)
    sentences = []
    for doc in docs:
        sentences.extend(nltk.sent_tokenize(doc.page_content))
    sentence_block = [" ".join(sentences[i: i+block_size]) for i in range(0, len(sentences), block_size)]
    nodes = [Node(properties={"page_content": sentence}) for sentence in sentence_block]
    return nodes

def add_extraction_to_nodes(nodes: List[Node], extractor) -> List[Node]:
    async def run_extractor(nodes):
        output = [extractor.extract(node) for node in nodes]
        extractor_output = await asyncio.gather(*output)
        return extractor_output

    extraction = asyncio.run(run_extractor(nodes))
    for (k,v), node in zip(extraction, nodes):
        v = list(map(lambda x: x.lower(), v))
        node.properties.update({k:v})
    return nodes 

def build_kg(nodes: List[Node], **kwargs) -> KnowledgeGraph:
    kg = KnowledgeGraph(nodes=nodes)
    rel_builder = JaccardSimilarityBuilder(property_name="entities", 
                                           threshold=kwargs.get("jaccard_threshold", 0.5),
                                           new_property_name="entity_jaccard_similarity")
    async def run_rel_builder(kg):
        return await rel_builder.transform(kg)
    kg_rel = asyncio.run(run_rel_builder(kg))
    for i, rel in enumerate(kg_rel):
        kg.add(rel)
    return kg

if __name__ == "__main__":

    pdf_path = "MRM_toolkit/data/wp_generative_ai_risk_management_in_fs.pdf"
    kg_path = "MRM_toolkit/KG/knowledge_graph.json"
    jaccard_threshold = 0.4

    page_split_from_mid = list(range(4,13))
    nodes = build_nodes_from_pdf(pdf_path=pdf_path, split_from_mid_page=page_split_from_mid)
    nodes = nodes[3:]

    print("Before extraction:")
    print([node.properties for node in nodes])
    extractor = NERExtractor()
    nodes = add_extraction_to_nodes(nodes, extractor)
    print("After extraction:")
    print([node.properties for node in nodes])
    kg = build_kg(nodes, 
                  jaccard_threshold=jaccard_threshold)
    print("Knowledge graph")
    print(kg)
    kg.save(path=kg_path)

