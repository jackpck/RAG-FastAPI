

if __name__ == "__main__":
    from ragas.testset.graph import KnowledgeGraph

    kg_path = "MRM_toolkit/KG/knowledge_graph.json"
    kg = KnowledgeGraph.load(kg_path)
    node0 = kg.nodes[0]
    print(node0.id)
    print("*"*20)
    print(node0.properties["page_content"])
    print("*"*20)
    rel0 = kg.relationships[0]
    print(rel0)
    print(rel0.source)
    print(rel0.target)
    print("")
    print(rel0.source.properties["page_content"])
    print("")
    print(rel0.source.properties["entities"])
    print("")
    print(rel0.target.properties["page_content"])
    print("")
    print(rel0.target.properties["entities"])