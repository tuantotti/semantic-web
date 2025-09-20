TEXT2CYPHER_PROMPT = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Do not use ** WHERE ** command in your cypher.

Let's consider examples:

Question: Tìm điểm dừng xe gần nhà tôi nhất (Đại học bách khoa Hà Nội)?
Output: 
```cypher 
MATCH (u:User)-[:REQUESTS]->(r:Route) 
MATCH (r)-[:HAS_STOP]->(s:Stop {{name: "Đại học bách khoa Hà Nội"}})  
RETURN r 
```

Now, generate cypher query for the following question:
Question: {question}
Output:"""

