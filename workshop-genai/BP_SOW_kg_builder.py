import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver
from neo4j_graphrag.generation.prompts import ERExtractionTemplate

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
   #model_name="gpt-5.1",
   model_name="gpt-4.1-mini",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

text_splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=50)

NODE_TYPES = [

    {"label": "Quote", "properties": [{"name": "id", "type": "STRING", "required": True}]},
  #  {"label": "Project", "properties": [{"name": "name", "type": "STRING", "required": True}]},
    {"label": "Customer", "properties": [{"name": "name", "type": "STRING", "required": True}]}
]
RELATIONSHIP_TYPES = [
    {"label": "QUOTE_FOR", },
]
PATTERNS = [
    ("Quote", "QUOTE_FOR", "Customer"),
    
]
writer = Neo4jWriter(driver=neo4j_driver, neo4j_database="neo4j", clean_db = True)

#print (ERExtractionTemplate.DEFAULT_TEMPLATE)
#prompt = '\nYou are a top-tier algorithm designed for extracting\ninformation in structured formats to build a knowledge graph.\n\nExtract the entities (nodes) and specify their type from the following text.\nAlso extract the relationships between these nodes.\n\nReturn result as JSON using the following format:\n{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],\n"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}\n\nUse only the following node and relationship types (if provided):\n{schema}\n\nAssign a unique ID (string) to each node, and reuse it to define relationships.\nDo respect the source and target node types for relationship and\nthe relationship direction.\n\nMake sure you adhere to the following rules to produce valid JSON objects:\n- Do not return any additional information other than the JSON in it.\n- Omit any backticks around the JSON - simply output the JSON on its own.\n- The JSON object must not wrapped into a list - it is its own JSON object.\n- Property names must be enclosed in double quotes\n\nExamples:\n{examples}\n\nQuote nofdes must only be created when explicitly mentioned and must have id starting with the letter Q followed by dash and numbers only. For example Q-1234\nn{text}\n'

prompt = '\nYou are a top-tier algorithm designed for extracting\ninformation in structured formats to build a knowledge graph.\n\nExtract the entities (nodes) and specify their type from the following text.\nAlso extract the relationships between these nodes.\n\nReturn result as JSON using the following format:\n{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],\n"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}\n\nUse only the following node and relationship types (if provided):\n{schema}\n\nAssign a unique ID (string) to each node, and reuse it to define relationships.\nDo respect the source and target node types for relationship and\nthe relationship direction.\n\nMake sure you adhere to the following rules to produce valid JSON objects:\n- Do not return any additional information other than the JSON in it.\n- Omit any backticks around the JSON - simply output the JSON on its own.\n- The JSON object must not wrapped into a list - it is its own JSON object.\n- Property names must be enclosed in double quotes\n\nExamples:\n{examples}\n'
#prompt = prompt + '\nQuote nodes must only be created when explicitly mentioned and must have id starting with the letter Q followed by dash and numbers only.\nFor example "Q-12345" will result in\n{{"nodes":[{{"id": "Q-12345", "label": "Quote"}}]}}\n'
prompt = prompt + '\nQuote nodes must only be created when a string starting with "Q-" and followed by digits is read\n'
prompt = prompt + '\nCustomer nodes must only be created when "Customer Name" is read\n'
prompt = prompt + '\n{text}\n'


kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    embedder=embedder, 
    from_pdf=True,
    text_splitter=text_splitter,
    schema={
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS,
        "additional_node_types": False,
    },
    prompt_template = prompt,
    kg_writer = writer,

)

pdf_file = "./workshop-genai/data/Bluestar PS BPO Enhancements Blue Planet Statement of Work.pdf"

result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)

id_resolver = SinglePropertyExactMatchResolver(driver=neo4j_driver, resolve_property ="id", neo4j_database=os.getenv("NEO4J_DATABASE"))
stats = asyncio.run(id_resolver.run())
print(stats)
name_resolver = SinglePropertyExactMatchResolver(driver=neo4j_driver, resolve_property ="name", neo4j_database=os.getenv("NEO4J_DATABASE"))
stats = asyncio.run(name_resolver.run())
print(stats)

