import os

import langchain_community.document_loaders
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from openai import azure_endpoint
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:
    folder_name = 'microwave_faiss_index'

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")
        # TODO:
        #  Check if `microwave_faiss_index` folder exists (os.path.exists("{folder_name}"))
        #  - Exists:
        #       It means that we have already converted data into vectors (embeddings), saved them in FAISS vector
        #       store and saved it locally to reuse it later.
        #       - Load FAISS vectorstore from local index (FAISS.load_local(...))
        #       - Configure folder_path `microwave_faiss_index`
        #       - Configure embeddings `self.embeddings`
        #       - Configure allow_dangerous_deserialization `True` (for our case it is ok, but don't do it on PROD)
        #       - Make variable assignment to `vectorstore`
        #  - Otherwise:
        #       - Make variable assignment of `self._create_new_index()` to `vectorstore`
        #  Return `vectorstore`
        if os.path.exists(self.folder_name):
            vectorstore = FAISS.load_local(folder_path=self.folder_name, embeddings=self.embeddings,
                                     allow_dangerous_deserialization=True)
        else:
            vectorstore = self._create_new_index()

        return vectorstore

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        loader = langchain_community.document_loaders.TextLoader(
            file_path='microwave_manual.txt',
            encoding='utf-8'
        )
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."])
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(
            documents = chunks,
            embedding=self.embeddings
        )
        vectorstore.save_local(self.folder_name)
        return vectorstore
        # TODO:
        #  1. Create langchain_community.document_loaders.TextLoader:
        #       - file_path is `microwave_manual.txt`
        #       - encoding is `utf-8`
        #       - assign it to `loader` variable
        #  2. Load documents via `loader.load()` and assign to `documents` variable
        #  3. Create RecursiveCharacterTextSplitter with
        #       - chunk_size=300
        #       - chunk_overlap=50
        #       - separators=["\n\n", "\n", "."]
        #  4. Split `documents` via created `text_splitter` into `chunks`
        #  5. Create `vectorstore` via FAISS.from_documents:
        #       - documents=chunks
        #       - embeddings=self.embeddings
        #  6. Save indexed data locally `vectorstore.save_local("microwave_faiss_index")`
        #  7. Return created `vectorstore`

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # TODO:
        #  Make `similarity_search_with_relevance_scores` in `vectorstore`:
        #       - query=query
        #       - k=k
        #       - score_threshold=score
        #       - assign results to `relevant_docs` variable
        relevant_docs = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )

        context_parts = []
        # TODO:
        #  Iterate through results (`for (doc, score) in relevant_docs`) and:
        #       - add `doc.page_content` to `context_parts` array
        #       - print `score`
        #       - print `page_content`

        for (doc, score) in relevant_docs:
            context_parts.append(doc.page_content)
            print(f"Page content : {doc.page_content}")
            print(f"Score : {score}")

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        response = self.llm_client.invoke(messages)
        print(f"{response.content}\n{'=' * 100}")

        # TODO:
        #  1. Create `messages` array with such messages:
        #       - SystemMessage(content=SYSTEM_PROMPT)
        #       - HumanMessage(content=augmented_prompt)
        #  2. Call self.llm_client.invoke(messages) and assign result to `response` variable
        #  3. print(f"{response.content}\n{'=' * 100}")
        #  4. Return response content
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        # Step 1: Retrieval
        context = rag.retrieve_context(user_question) # Here you can play with `k` and similarity score params
        # Step 2: Augmentation
        augmented_prompt = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        answer = rag.generate_answer(augmented_prompt)


main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment='text-embedding-3-small-1',
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY)
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment='gpt-4o',
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version=""
        )
        # TODO:
        #  1. pass embeddings:
        #       - AzureOpenAIEmbeddings
        #       - deployment='text-embedding-3-small-1'
        #       - azure_endpoint=DIAL_URL
        #       - api_key=SecretStr(API_KEY)
        #  2. pass llm_client:
        #       - AzureChatOpenAI
        #       - temperature=0.0
        #       - azure_deployment='gpt-4o',
        #       - azure_endpoint=DIAL_URL
        #       - api_key=SecretStr(API_KEY)
        #       - api_version=""
    )
)