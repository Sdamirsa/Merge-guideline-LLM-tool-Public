new_summary_tmpl_str = (
    "You are an assistant for question-answering tasks.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "Use three sentences minimum and keep the answer concise.\n"
    "If the context for query mentions the level of evidence of that information please say that too.\n"
    "If you don't know the answer, just say that you don't know.\n"
    "if the query questions answer have not been provided in the context just say the given title and say it does not been mentioned in the given text. \n"
    "Context information is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this  context information and not prior knowledge, answer the given query: \n"
    "Query: {query_str}\n"
    "Answer: "
)

title_generation_prompt = '''your a title generator system. based on a document summary you should generate keyword based titles of what is spoken in a article.\n you should generate as much as titles as you can, so when a user look at titles they can imagine the paper structure and what is happend on paper.\n please just provide this and nothing more.\n generate titles by comma seperation. for example: title_1, title2. 
'''

similarity_prompt = '''
your a similarity and disimilarity generator system. based on a section of different medical documents on the same title and the same topic you should point out similiar opinions of this douments and thier differencess as well.\n.the user provides the file names for you, use them in writing differences\n please just provide this and nothing more.\n please write differences for each document and if document does not have difference with others say that.\n please say the differences in a way that the contrast is visible for the user and can easily understand the differences between documents \n please say the differences in bullet points \n generate it as this format:\n
    **similarities**:\n
    + all documents say that:<similarity one>\n
    + all documents say that:<similarity two>\n
    **differences**:\n
    - <document name> says: <difference in bullet points>\n
    - <document name2> says: <difference in bullet points>\n
    **final conclusion**:
    based on all documents opinions please provide a consice medical scientific conclusion about the question the documents try to answer. the question is usually in the first line of users prompt.
'''

ultimate_outline_prompt = '''
you are an academic table of contents generator. you will be given one or more refrences and thier titles. \n
you goal is to make an ultimate table of contents from this given text. \n
you should order it and make an orders table of contents like a well written book or academic paper. \n
please for making the readability better make headings and subheadings for better understanding and readability. \n
i want a merged outline of all documents dont just seperate them by document names, use all of socuments titles together to make a complete table of content like scientific journals or books. \n
this is the desiered output:\n
# an important topic \n
## an importatnt sub-topic of that topic\n
### an importatnt sub-topic of that sub-topic\n
## anohter importatnt sub-topic of that topic\n
# another important topic\n\n
please just ouput this, i want to develope a system that ignores anything that dont follow this format so please obey this format. 
'''

###########################################################################################
# importing part
#########################################################################################
import streamlit as st # type: ignore

from openai import OpenAI as OAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext 
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor,KeywordExtractor,SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.postprocessor import SimilarityPostprocessor,SentenceEmbeddingOptimizer
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core import get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.query_engine import MultiStepQueryEngine


from dotenv import load_dotenv, set_key, find_dotenv
import os
import pickle
# import nest_asyncio
# nest_asyncio.apply()


##########################################################################################
# definitions part
#########################################################################################
folder_path = './articles'
summary_db_path = './store.pkl'
cache_path = './cache.pkl'
storage_file_path = './storage'
chunk_size=3000
chunk_overlap=600

new_summary_tmpl_str = constants.new_summary_tmpl_str
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)


###########################################################################################
# functions part
#########################################################################################

# some how it doesnt work. please check that.
def standradize_doc_ids(documents):
    """
    Standardizes the IDs of documents in st.session_state.documents.
    
    This function iterates over each document in st.session_state.documents, 
    and updates its ID by combining the filename (without the '.pdf' extension) 
    with the page label. The resulting ID is in the format 'filename_page_label'.

    For example, if a document's filename is 'example.pdf' and its page label is 1, 
    its ID would be updated to 'example_1'.
    """
    for doc in documents:
        print(doc.id_)
        # print(doc.metadata["file_name"])
        # print(doc.metadata["page_label"])
        b = doc.metadata["file_name"].split(".pdf")[0]
        c = b + "_" + str(doc.metadata["page_label"])
        doc.id_ = c
        print(doc.id_)
    return documents

def check_new_documents(documents):
    """
    Checks if there are new documents added to the directory.

    This function takes a list of documents as input and returns a list of new documents
    that do not already exist in the index storage context.

    Args:
        documents (list): A list of Document objects to check for new additions.

    Returns:
        list: A list of new Document objects that were not previously indexed.

    """
    new_docs = []
    for doc in documents:
        # Check if the document with the current ID does not already exist in the index storage context.
        # If it doesn't exist, it's a new document that needs to be processed.
        if not st.session_state.index.storage_context.docstore.ref_doc_exists(doc.id_):
            print(doc.id_)
            print(st.session_state.index.storage_context.docstore.ref_doc_exists(doc.id_))
            new_docs.append(doc)
    return new_docs



def extract_file_names(folder_path: str) -> list[str]:
    """
    Extracts a list of file names from a specified folder path.

    Args:
        folder_path (str): The path to the folder from which to extract file names.

    Returns:
        list[str]: A list of file names found in the specified folder.

    Raises:
        FileNotFoundError: If the folder path does not exist.

    Notes:
        This function iterates over all files in the specified folder and returns a list of their names.
    """
    try:
        # Use os.listdir to get a list of all files and directories in the folder
        files_and_dirs = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"The folder {folder_path} does not exist.")
        return []

    # Use a list comprehension to filter out directories and return a list of file names
    file_names = [file_name for file_name in files_and_dirs if os.path.isfile(os.path.join(folder_path, file_name))]

    print(f"File names have been extracted. {len(file_names)} files found.")
    return file_names





def check_summary_db_exists(summary_db_path):
    '''
    summary db contains: 
    
    '''
    if os.path.exists(summary_db_path):
        with open(summary_db_path, "rb") as f:
            summary_db = pickle.load(f)
    else:
        summary_db = {}
    return summary_db

def run_summary_indexer(documents):
    """
    Creates a Document Summary Index from a list of documents.

    This function initializes a response synthesizer and a storage context for 
    storing document summaries. Then, it generates a DocumentSummaryIndex from 
    the provided documents using the specified LLM and transformations.

    Args:
        documents (list): A list of Document objects for which summaries 
                          should be created.

    Returns:
        DocumentSummaryIndex: An instance of DocumentSummaryIndex containing 
                              the summaries of the provided documents.
    """
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", use_async=True)

    storage_context_summary = StorageContext.from_defaults()

    doc_summary_index = DocumentSummaryIndex.from_documents(
        storage_context= storage_context_summary,
        documents=documents,
        llm=st.session_state.openai_llm,
        transformations=[st.session_state.sen_splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )
    return doc_summary_index

# def merge_document_summaries(documents):
#     """
#     Merges the summaries of documents into a dictionary based on their filenames.

#     This function iterates over a list of Document objects and generates a summary for 
#     each document using the session state's Document Summary Index. The summaries are 
#     grouped by the filename (from the document's metadata), and the resulting dictionary 
#     contains filenames as keys and concatenated summaries as values.

#     Args:
#         documents (list): A list of Document objects, each containing metadata with 
#                           'file_name' and an ID for retrieving summaries.

#     Returns:
#         dict: A dictionary where the keys are filenames and the values are strings 
#               containing the merged summaries for each corresponding document.
#     """
#     a_dict = {}
#     for doc in documents:
        
#         if doc.metadata['file_name'] not in a_dict.keys():
#             tr = doc.metadata['file_name']
#             a_dict[tr] = []
#             summ = st.session_state.doc_summary_index.get_document_summary(doc.id_)
#             a_dict[tr].append(summ)
#         else:
#             tr = doc.metadata['file_name']
#             summ = st.session_state.doc_summary_index.get_document_summary(doc.id_)
#             a_dict[tr].append(summ)
#     for k,v in a_dict.items():
#         big_string = ' '.join(v)
#         a_dict[k]= big_string

#     return a_dict


def merge_document_summaries(documents):
    """
    Merges the summaries of documents into a dictionary based on their filenames.

    This function iterates over a list of Document objects and generates a summary for 
    each document using the session state's Document Summary Index. The summaries are 
    grouped by the filename (from the document's metadata), and the resulting dictionary 
    contains filenames as keys and concatenated summaries as values.

    Args:
        documents (list): A list of Document objects, each containing metadata with 
                          'file_name' and an ID for retrieving summaries.

    Returns:
        dict: A dictionary where the keys are filenames and the values are strings 
              containing the merged summaries for each corresponding document.
    """
    summaries_by_filename = {}
    
    for doc in documents:
        filename = doc.metadata['file_name']
        summ = st.session_state.doc_summary_index.get_document_summary(doc.id_)

        if filename not in summaries_by_filename:
            summaries_by_filename[filename] = []

        summaries_by_filename[filename].append(summ)

    # Merge summaries into single strings for each filename
    for filename, summaries in summaries_by_filename.items():
        summaries_by_filename[filename] = ' '.join(summaries)

    return summaries_by_filename

# def generate_title_response(dictio):
#     title_response = {}
#     for k,v in dictio.items():
#         messages = [
#         ChatMessage(
#             role="system", content=constants.title_generation_prompt
#         ),
#         ChatMessage(role="user", content= "there is a document summary: \n" + v ),
#         ]
#         resp = OpenAI(model="gpt-4o-2024-08-06").chat(messages)
#         titles_list = [title.strip() for title in resp.message.content.split(',')]
#         title_response[k] = titles_list
#     return title_response

def generate_titles_for_summary(summary):
    """
    Generates titles for a given document summary using a chat model.

    This function sends a structured message to a language model, specifying that 
    it is tasked with generating titles based on the provided document summary. 
    The function prepares the interaction by defining the roles (system and user) 
    and then processes the response to extract and format the generated titles.

    Args:
        summary (str): A string containing the document summary for which titles 
                       are to be generated.

    Returns:
        list: A list of titles extracted from the model's response, each title 
              stripped of leading and trailing whitespace.
    """

    messages = [
            ChatMessage(role="system", content=constants.title_generation_prompt),
            ChatMessage(role="user", content=f"there is a document summary: \n{summary}"),
        ]
    response = OpenAI(model="gpt-4o").chat(messages)
    return [title.strip() for title in response.message.content.split(',')]


def generate_title_response(document_dict):
    """
    Generates titles for a dictionary of document summaries.

    This function takes a dictionary where the keys are document IDs and the values 
    are corresponding summaries. For each document summary, it invokes the function 
    `generate_titles_for_summary` to create a list of titles. The resulting dictionary 
    contains the same document IDs as keys and the generated titles as values.

    Args:
        document_dict (dict): A dictionary where keys are document IDs and values 
                              are strings representing document summaries.

    Returns:
        dict: A dictionary where keys are document IDs and values are lists of
              titles generated based on the corresponding summaries.
    """
    title_response = {}
    for document_id, summary in document_dict.items():
        title_response[document_id] = generate_titles_for_summary(summary)

    return title_response


def find_unique_elements(list1, list2):
    """
    Finds unique elements in two lists.

    This function takes two lists as input and determines which elements are unique to each list.
    It converts the lists to sets to efficiently perform set operations, and then identifies the
    elements that are in one list but not the other. The result combines these unique elements 
    into a single list.

    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.

    Returns:
        list: A list of unique elements that are found in either list1 or list2, 
              excluding elements present in both.
    """
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    
    # Find unique elements in both sets
    unique_in_list1 = set1 - set2  # Elements in list1 but not in list2
    unique_in_list2 = set2 - set1  # Elements in list2 but not in list1
    
    # Combine the unique elements into a single list
    unique_elements = list(unique_in_list1.union(unique_in_list2))
    
    return unique_elements


def transform_dict_to_string(file_dict):
    """
    This function takes a dictionary of file names and their corresponding text as input.
    It then transforms this dictionary into a single string, where each file's text is 
    prefixed with a description of the file.
    
    this is for the part that for the query each question asked seperately for each document.
    this function make the dictionary of each file name and retrived chunks to a mega text suitabale for LLM ingestion. 
    
    """
    result = ""
    for file_name, file_text in file_dict.items():
        result += f'**the docuemnt "{file_name}" says**:\n \n {file_text} \n \n'
    return result.strip()

def on_api_key_change():
    dotenv_path = find_dotenv()
    set_key(dotenv_path, "OPENAI_API_KEY", st.session_state['api_key'])
    st.session_state['api_key'] = ""
    st.session_state.oai_key_status = True
###########################################################################################

# streamlit part

#########################################################################################


if "title_status" not in st.session_state:
    st.session_state.title_status = False

if "login_status" not in st.session_state:
    st.session_state.login_status = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello!"})
if "oai_key_status" not in st.session_state:
    st.session_state.oai_key_status = False
    
####################################################
##rfacotr part
##list of functions now: ["standradize_doc_ids", "check_new_documents", "extract_file_names", "check_summary_db_exists", "run_summary_indexer", "merge_document_summaries", "generate_titles_for_summary", "generate_title_response", "find_unique_elements", "transform_dict_to_string"]

## list of st.sessions : ["title_status","login_status","documents",'file_names','storage_context','index','titled_response','doc_summary_index','title_status','login_status','messages','openai_model']

def show_st_intro_text():
        st.header("Log in")

        st.markdown("> please upload your documents in **articles folder**")
        st.markdown("> if you have uploaded your documents for the first time or you uploaded new documents, click on **LOAD DOCUMENTS** button")
        st.markdown("> if you loaded your documents before or have loaded now, click on **show titles** button to get a gist of the documents you want to request before you log in")
        st.markdown("> if you loaded your documents before or have loaded now, click on **Log in** button")

def load_documents_to_st():
    documents = SimpleDirectoryReader('./articles').load_data()
    documents = standradize_doc_ids(documents)
    st.session_state.documents = documents
    file_names = extract_file_names(folder_path)
    st.session_state.file_names = file_names

def load_database_from_memory(storage_file_path):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    st.session_state.storage_context = storage_context
    index = load_index_from_storage(storage_context)
    st.session_state.index = index
    return storage_context,index
    
def proccess_new_documents(new_docs):
    print("👉 Adding new documents: ", len(new_docs))
    text = "👉 Adding new documents: " + str(len(new_docs))
    st.markdown(text)
    nodes = st.session_state.pipeline.run(documents=new_docs)
    st.session_state.index.insert_nodes(nodes)
    st.session_state.index.storage_context.persist(persist_dir="./storage")

def run_database_for_first_time():
    text = "the programm is running for the first time. adding " + str(len(st.session_state.documents)) + " docs"
    print(text)
    st.markdown(text)
    nodes = st.session_state.pipeline.run(documents=st.session_state.documents)
    storage_context = StorageContext.from_defaults()
    st.session_state.storage_context = storage_context
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir="./storage")
    st.session_state.index = index

def run_show_titles_first_time():
    
    merged_document_summaries = check_summary_db_exists(cache_path)
    if bool(merged_document_summaries):
        print("using cache ....")
    else:
        print("not using cache ....")
        # doc_summary_index = run_summary_indexer(st.session_state.documents)
        # st.session_state.doc_summary_index = doc_summary_index
        
        # merged_document_summaries = merge_document_summaries(st.session_state.documents)
        # with open("cache.pkl", "wb") as f:
        #     pickle.dump(merged_document_summaries, f)
    #A dictionary where keys are document IDs and values are lists of titles generated based on the corresponding summaries.
    titled_response = generate_title_response(merged_document_summaries)
    st.session_state.titled_response = titled_response
    with open("store.pkl", "wb") as f:
        pickle.dump(titled_response, f)


def show_titles_in_st_app():
                pdf_titles = st.session_state.titled_response
                for pdf_name, titles in pdf_titles.items():
                    with st.expander(pdf_name):
                        for title in titles:
                            st.markdown(f"- {title}")

def show_chat_messages_in_st():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def user_prompt_proccess(prompt):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

def run_rag_query(file_name,prompt):
    """Run RAG query for a given file."""
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="file_name", value=file_name)]
    )
    
    query_engine = st.session_state.index.as_query_engine(
        filters=filters,
        similarity_top_k=4,
        response_mode="tree_summarize",
    )
    query_engine.update_prompts(
        {"response_synthesizer:summary_template": new_summary_tmpl}
    )
    return query_engine.query(prompt)


def process_query_results(file_name, query_results):
    """Process the results of the RAG query."""
    text_sources = [uii.get_text() for uii in query_results.source_nodes]
    response_summary = query_results.response
    return text_sources, response_summary

def proccess_final_response(query,sim_query_dict):
    #formatted string is like: "document_1" says: says that.
    formatted_string = 'the question these documents are trying to answer is: ' + query + '\n \n'
    formatted_string =  formatted_string+transform_dict_to_string(sim_query_dict)
    messages = [
        ChatMessage(
            role="system", content=constants.similarity_prompt),
        ChatMessage(role="user", content= formatted_string),
        ]
    resp = OpenAI(model=st.session_state["openai_model"]).chat(messages)
    ultimate_text = formatted_string  + "\n \n" + resp.message.content
    response = st.markdown(ultimate_text)
    st.session_state.messages.append({"role": "assistant", "content": response})

def make_ultimate_outline():
    ultimate_outline_raw = ''''''
    pdf_titles = st.session_state.titled_response
    # print(pdf_titles)
    # print('-------------')
    for pdf_name, titles in pdf_titles.items():
            ultimate_outline_raw += '"'+pdf_name + '" : '
            ultimate_outline_raw += ', '.join(titles)
            ultimate_outline_raw += '. \n \n'
    messages = [
        ChatMessage(role="system", content=constants.ultimate_outline_prompt),
        ChatMessage(role="user", content= ultimate_outline_raw),]
    print(ultimate_outline_raw)
    resp = OpenAI(model=st.session_state["openai_model"],timeout=300,max_retries=1).chat(messages)
    print(resp.message.content)
    mark_downed = convert_to_markdown(resp.message.content)
    st.session_state.mark_downed =  mark_downed

def convert_to_markdown(text):
    # Split the text into lines
    lines = text.strip().split('\n')
    
    markdown_content = []
    indentation_level = 0
    previous_level = 0
    header_prefixes = {
        '#': 0,   # Level 0 for #
        '##': 1,  # Level 1 for ##
        '###': 2  # Level 2 for ###
        ,'####': 3  
        ,'#####': 4  
        ,'######': 5  
        }
                    
    for line in lines:
        stripped_line = line.strip()
        
        # Determine the header level by counting the '#' at the start
        if stripped_line.startswith('#'):
            prefix_length = len(stripped_line.split(' ', 1)[0])  # Count the number of '#'
            indentation_level = header_prefixes.get('#' * prefix_length, 0)
            # Remove '#' and leading/trailing spaces for the markdown content
            content = stripped_line.lstrip('#').strip()
            # Indentation spaces for markdown style
            markdown_prefix = '  ' * indentation_level + '- ' 
            markdown_content.append(markdown_prefix + content)
        else:
            # If it's not a header, just append the text content as it is
            markdown_content.append('  ' * (indentation_level + 1) + stripped_line)
    
    # Join the markdown lines to form final output
    return '\n'.join(markdown_content)
####################################################
#the first page of streamlit app wrapped in function
def login():
    load_dotenv()

    # global llamaindex Settings
    st.session_state.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    st.session_state.openai_llm = OpenAI(model="gpt-4o",temperature=0.7)
    Settings.embed_model = st.session_state.embed_model
    Settings.llm = st.session_state.openai_llm


    # defining ingestion pipeline
    sen_splitter =SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    st.session_state.sen_splitter = sen_splitter

    pipeline = IngestionPipeline(
        transformations=[
            sen_splitter,
            SummaryExtractor(summaries=["prev", "self", "next"]),
            KeywordExtractor(keywords=10,),
        ]
    )
    st.session_state.pipeline = pipeline
    show_st_intro_text()
    #difining columns for the buttons of app.     
    col2, col3, col4 = st.columns(3)
    ###########################################################################
    with col2:
        if st.button("load documents"):
            load_documents_to_st()
            # after loading documents we check if thedatabse and indexer for proccessing documents are exist beforehand.

            # if database exist from before
            if os.path.exists(storage_file_path):

                storage_context,index = load_database_from_memory(storage_file_path)

                #it returns list of new documnets
                new_docs = check_new_documents(documents=st.session_state.documents)
                if new_docs:
                    proccess_new_documents(new_docs=new_docs)
                else:
                    st.markdown("No new documents to add")

            # the database dose not exist and program is running for the first time
            else:
                run_database_for_first_time()
    #############################################################################            
    # runs the summerizer part which generates titles
    with col3:
        if st.button("show titles"):
            #check if documents exist from step before
            if "documents" not in st.session_state:
                load_documents_to_st()

            #check if a database exsists: either returns empty {} or a dictionary from before
            summary_db = check_summary_db_exists(summary_db_path)


            #if database full and no new ducoments: show titles
            #if database full but new docuemts: add them first then show titles
            if bool(summary_db):
                articles_have_summary_list = list(summary_db.keys())
                all_article_lists = st.session_state.file_names
                articles_dont_have_summary = find_unique_elements(articles_have_summary_list, all_article_lists)
                if len(articles_dont_have_summary) == 0:
                    st.markdown("you DONT have new documents to proccess")
                    st.session_state.titled_response = summary_db
                else:
                    st.markdown("you have new documents to proccess")
                    st.session_state.titled_response = summary_db
                    #TODO complete this

            #if no database: add the documents 
            if not bool(summary_db):
                run_show_titles_first_time()
            st.session_state.title_status = True
    
    ############################################################################################
    # changes the page  to the llm page
    with col4:
        if st.button("Log in"):
            st.session_state.login_status = True
            st.rerun()
    
    # show titles and ultimate titles if there were availabale
    if st.session_state.title_status:
        if "titled_response" in st.session_state:
            show_titles_in_st_app()
            if st.button("make the ULTIMATE outline."):
                make_ultimate_outline()
                
        if "mark_downed" in st.session_state:
            with st.expander("Ultimate outline"):
                    st.markdown(st.session_state.mark_downed)
            st.download_button(
                label='Download Markdown File',
                data=st.session_state.mark_downed,
                file_name='ultimate_outline.md',
                mime='text/markdown'
            )


def llm():
    load_dotenv()

    # global llamaindex Settings
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    openai_llm = OpenAI(model="gpt-4o", temperature=0.7)
    Settings.embed_model = embed_model
    Settings.llm = openai_llm

    show_chat_messages_in_st()

    if prompt := st.chat_input("What is up?"):
        
        user_prompt_proccess(prompt=prompt)

        sim_query_dict = {}
        sim_query_cite = {}
        
        # Run RAG query for each document
        for file_name in st.session_state.file_names:
            sim_query_cite[file_name] = []
            
            query_results = run_rag_query(file_name=file_name, prompt=prompt)
            text_sources, response_summary = process_query_results(file_name, query_results)
            sim_query_cite[file_name].extend(text_sources)

            sim_query_dict[file_name] = response_summary
            print('For this file name:', file_name)
            print(response_summary)

        # Display sim_query_cite in the sidebar
        with st.sidebar:
            st.write("Retrieved Texts by File")
            for file, texts in sim_query_cite.items():
                st.write(f"**{file}**")
                for index,text in enumerate(texts):
                    st.write(f"some of the text of the chunk_{index}: \n \n {text[:300]}\n \n")

        # Transforms the dictionary of each file retrieved text to mega text to feed to LLM for final response
        proccess_final_response(query=prompt, sim_query_dict=sim_query_dict)
        
        # Save sim_query_cite to a file
        with open(f"test.pkl", "wb") as mamad:
            pickle.dump(sim_query_cite, mamad)


def key():
    st.write("set your OpenAI API key before proceeding, if you dont have one please go to https://platform.openai.com/account/api-keys")
    st.text_input('OpenAI API key', type='password', key='api_key',  label_visibility="collapsed")
    st.button("set as env variable",on_click=on_api_key_change,)

st.title("similarity? disimilarity? find out with Us :)")

if st.session_state.login_status:
    pg = st.navigation([st.Page(llm)])    
else:
    if st.session_state.oai_key_status:
        pg = st.navigation([st.Page(login)])
    else:
        pg = st.navigation([st.Page(key)])

pg.run()
