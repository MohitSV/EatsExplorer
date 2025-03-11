import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import pandas as pd
import uuid

# Read the CSV file into a DataFrame
df = pd.read_csv('menudata.csv')


# unique_sorted = sorted(df['state'].dropna().unique())
# chunks = [unique_sorted[i:i + 20] for i in range(0, len(unique_sorted), 20)]

#Convert each chunk to strings before joining
# chunks_as_strings = [', '.join(map(str, chunk)) for chunk in chunks]
chunks_as_strings = [[94110]]
# Print each chunk on a new line
documents = []
for chunk in chunks_as_strings:

    documents.append(
        Document(
            page_content=chunk,
            metadata={
                "column": "zip_code"
            }
        )
    )

index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("zip_code")))
print(index)
vector_store = FAISS(
    embedding_function=OpenAIEmbeddings(),
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)

# import os
# os.makedirs("faiss_index", exist_ok=True)


vector_store.add_documents(documents=documents, ids=[i for i in range(len(documents))])

vector_store.save_local("zip_code")

embeddings = OpenAIEmbeddings()

vector_store_list = [
    "address1",
    "categories",
    "menu_item",
    "menu_category",
    "menu_description",
    "restaurant_name",
]

# Initialize empty documents list
documents = []
document_ids = []

# Create documents for each column
for column in vector_store_list:
    # Get unique values from the column, sort them, and drop NA values
    unique_sorted = sorted(df[column].dropna().unique())
    
    # Chunk into groups of 20
    chunks = [unique_sorted[i:i + 20] for i in range(0, len(unique_sorted), 20)]
    
    # Create a document for each chunk
    for chunk in chunks:
        # Join the chunk values with commas
        chunk_content = ', '.join(str(value) for value in chunk)
        documents.append(
            Document(
                page_content=chunk_content,
                metadata={
                    "column": column
                }
            )
        )
        document_ids.append(str(uuid.uuid4()))

# Create the vector store
embedding_size = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(embedding_size)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add all documents to the vector store with UUID4 ids
vector_store.add_documents(documents=documents, ids=document_ids)

# Save the combined vector store
vector_store.save_local("menudata")

docs = new_vector_store.similarity_search_with_score('Courtland Ave', k=3, filter={"column": "address1"})
for doc in docs:
    print(doc)
    print("\n")

# Load the saved vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("menudata", embeddings, allow_dangerous_deserialization=True)

# Perform similarity search
# You can filter by column and specify number of results with k
results = vector_store.similarity_search_with_score(
    query="courtland ave",  # Replace with your search query
    k=3,  # Number of results to return
    filter={"column": "address1"}  # Filter by specific column
)
print(results)
# Print results
for doc, score in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print("\n")



