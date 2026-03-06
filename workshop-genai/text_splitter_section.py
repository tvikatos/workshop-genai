import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
import pdfplumber
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.types import DocumentInfo, LexicalGraphConfig
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder

# tag::import_splitter[]
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
# end::import_splitter[]

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
#neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# tag::splitter[]
class SectionSplitter(TextSplitter):
    def __init__(self, section_heading: str = "== ") -> None:
        self.section_heading = section_heading

    async def run(self, text: str) -> TextChunks:
        index = 0
        chunks = []
        current_section = ""

        for line in text.split('\n'):
            # Does the line start with the section heading?
            if line.startswith(self.section_heading):
                chunks.append(
                    TextChunk(text=current_section, index=index)
                )
                current_section = ""
                index += 1
            
            current_section += line + "\n"

        # Add the last section
        chunks.append(
            TextChunk(text=current_section, index=index)
        )
        
        return TextChunks(chunks=chunks)
    

def get_chunks_from_pages(pdf:pdfplumber.pdf):
    index = 0
    chunks = []
    for page in pdf.pages:
        #print("Index ",index,"\n",page.extract_text())
        chunks.append(
            TextChunk(text = page.extract_text(), index = index)
        )
                
        index += 1
    return TextChunks(chunks=chunks)

#splitter = SectionSplitter()
text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)
# end::splitter[]

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
            #print (page.page_number," -- ",page.extract_text(),"\n")
    return text

# tag::run_splitter[]


#===============================
# gets text out of pdf, splits it in chunks (one per page), creates embeddings using LLM, creates graph and stores in db
#------------------------------------
pdf_file = "./workshop-genai/data/Bluestar PS BPO Enhancements Blue Planet Statement of Work.pdf"
#text = extract_text_from_pdf(pdf_file)
pdf = pdfplumber.open(pdf_file)
#chunks = asyncio.run(text_splitter.run(text))
chunks = get_chunks_from_pages(pdf)
chunks_embedder = TextChunkEmbedder(embedder);
chunks = asyncio.run(chunks_embedder.run(text_chunks = chunks))
#print(chunks)

doc = DocumentInfo(path = pdf_file)
config=LexicalGraphConfig(id_prefix='', 
                                document_node_label='SOW', 
                                chunk_node_label='Chunk', chunk_to_document_relationship_type='FROM_DOCUMENT', 
                                next_chunk_relationship_type='NEXT_CHUNK', 
                                node_to_chunk_relationship_type='FROM_CHUNK', 
                                chunk_id_property='id', 
                                chunk_index_property='index', 
                                chunk_text_property='text', 
                                chunk_embedding_property='embedding'
                                )

lex_builder =LexicalGraphBuilder(config)

gresult = asyncio.run(lex_builder.run(text_chunks = chunks, document_info=doc))
#print("LEXBUILDER RESULT: \n",gresult)

#graph = Neo4jGraph(input_value = gresult.graph)
#print("GRAPH IS: \n",graph)

writer = Neo4jWriter(driver=neo4j_driver, neo4j_database="neo4j", clean_db = True)
print(asyncio.run(writer.run(graph=gresult.graph, lexical_graph_config=config)))

#==========================================================================================

"""
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder,
    from_pdf=True,
    text_splitter=splitter,
)
# end::kg_builder[]



print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
"""
