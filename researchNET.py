import autogen
import requests
import os
import arxiv
import datetime
from typing import List, Dict, Optional
from docx import Document
from autogen import OpenAIWrapper
from langchain_community.tools import PubmedQueryRun

config_file_or_env = "OAI_CONFIG_LIST"

config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["azure"]})
openai_wrapper = OpenAIWrapper(config_list=config_list)

config_list_turbo = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-3.5-turbo-1106"]})
#openai_wrapper_turbo = OpenAIWrapper(config_list=config_list_turbo)

llm_config = {
    "cache_seed": 50,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

fastllm = {
    "cache_seed": 49,
    "temperature": 0,
    "config_list": config_list_turbo,
    "timeout": 120,
}

def download_files(urls: list) -> str:
    """
    Downloads multiple files from the specified URLs and saves them as PDFs in the script's current working directory,
    using their original names but ensuring they have a .pdf extension.

    :param urls: A list of URLs of the files to download.
    """
    # Get the directory of the currently executing script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    for url in urls:
        try:
            # Extracting filename from URL
            original_filename = os.path.basename(url)
            # Ensuring the saved file has a .pdf extension
            pdf_filename = f"{original_filename}.pdf"

            # Construct the full path where the file will be saved
            destination = os.path.join(script_directory, pdf_filename)

            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raises an HTTPError if the response was an error

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"File '{original_filename}' downloaded successfully and saved as '{pdf_filename}' to {destination}")
        except Exception as e:
            print(f"Failed to download '{url}': {e}")
    return "Downloaded files successfully."

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A human admin. Interact with the research_executive to discuss the plan. Plan execution needs to be approved by this admin.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)

research_executive = autogen.AssistantAgent(
    name="research_executive",
    llm_config=llm_config,
    system_message="""You are the research_executive in a group chat focused on completing a research task given by the user. Your role is to oversee the research team, ensuring quality and workflow efficiency. Your key responsibilities include:
    - Think about the task the user gave you and decide what is the most efficent and high quality way to solve it.
    - Collaborate with scientific_researcher for paper gathering, literature_reader for paper analysis, scientific_writer for drafting text, and scientific_editor for content refinement.
    - The scientific_editor will give you the final draft for your final oversight, look critically over it and decide if it fits the request from the user, otherwise delegate it back to the team for correction.
    - If the users request HAS been addressed, respond with the final draft and save it as .docx file.
    - The final draft MUST end with the word TERMINATE. If the  user request is pleasantry or greeting, you should respond with a pleasantry or greeting and TERMINATE.""",
    description="""A research_executive is the first and the last in the group workflow. He plans the task given by the input from the user, coordinates the research team, guiding task delegation, ensuring quality, providing feedback, and signaling task completion to achieve research objectives efficiently."""
)

scientific_researcher = autogen.AssistantAgent(
    name="scientific_researcher",
    llm_config=llm_config,
    system_message="""As a scientific_researcher in a group chat, your expertise lies in exploring scientific databases and identifying relevant literature. Your key responsibilities include:
    - Formulate relevant keywords for the best search results in scientific databases like arXiv, pubMed or Google Scholar.
    - Use your ability to search in scientific databases for proper papers.
    - Provide the found literature without changing the output to the literature_reader.
    - The literature_reader can give you orders to find additional literature if he is not satisfied with your results.
    If uncertainties arise or further assistance is needed, reach out to the research_executive for guidance or redirection.""",
    description="""A scientific_researcher specializes in efficiently providing relevant scientific literature from databases like PubMed and arXiv, tailored to the research team's needs."""
)

literature_reader = autogen.AssistantAgent(
    name="literature_reader",
    llm_config=llm_config,
    system_message="""As a literature_reader in a group chat, your primary task is to analyze academic papers, distill key points, and assign a relevance score from 0 to 10, reflecting each material's importance to the research topic. Your key responsibilities include:
    - Critically evaluate academic literature found by the literature_reader and summarize key findings.
    - Rate the relevance of each piece of literature to the provided research topic.
    - Request aditional literature from the scientific_researcher if you are not satisfied with the provided literature.
    - If you are satisfied with the literature and think the writing task for the given research topic can be completed with it, delegate it to the Scientifc Writer.
    If uncertainties arise or further assistance is needed, reach out to the research_executivee for guidance or redirection.""",
    description="""A literature_reader critically evaluates and summarizes academic papers found by the scientific_researcher, assigning relevance scores to guide further research analysis and requesting additional literature as needed."""
)

scientific_writer = autogen.AssistantAgent(
    name="scientific_writer",
    llm_config=llm_config,
    system_message="""As a scientific_writer in a group chat, you are tasked with producing written content that integrates findings from scientific literature. Your key responsibilities include:
    - Drafting a manuscript for the given research topic using literature provided by the literature_reader.
    - Writing in a concise an scientific writing style.
    - Counting the characters of the manuscript to reach the goal provided by the research_executive.
    - Cite the used literature in APA format and organize them into a Bibliography Section.
    - When your draft is completed, give it to the scientific_editor for review.
    - Refine drafts in collaboration with the Editor based on feedback.
    If uncertainties arise or corrections are needed, consult the research_executive for support or to address discrepancies within the chat.""",
    description="""A scientific_writer crafts comprehensive manuscripts based on provided literature, ensuring character count goals are met with proper citation, and collaborates with the scientific_editor for refinement."""
)

scientific_editor = autogen.AssistantAgent(
    name="scientific_editor",
    llm_config=llm_config,
    system_message="""As the scientific_editor in a group chat, your primary duty is to refine and validate the content produced by the scientific_writer. Your key responsibilities include:
    - Review the content from the scientific_writer for clarity, coherence, and scientific accuracy, ensuring it aligns with the research topic and is proper cited according to the literature and bibliography section.
    - If you are not satisfied provide detailed feedback for the scientific_writer to improve the scientific narrative's quality.
    - If the manuscript is ready for final submissions give it to the research_executive for final review.
    If uncertainties arise or corrections are needed, consult the research_executive for support or to address discrepancies within the chat.""",
    description="""A scientific_editor critically reviews and refines manuscripts from the scientific_writer for clarity, coherence, and accuracy, providing feedback for improvements and preparing content for final evaluation by the Research Executive."""
)

#@user_proxy.register_for_execution()
#@scientific_researcher.register_for_llm(description="searches pubMed for papers based on a given query string and caches the results.")
def pubmed_search(query: str) -> str:
    # Create an instance of PubmedQueryRun
    tool = PubmedQueryRun()
  
    # Run the search with the specified query
    search_result = tool.run(query)
  
    # Return the search result
    return search_result

@user_proxy.register_for_execution()
@scientific_researcher.register_for_llm(description="searches arXiv for papers based on a given query string and caches the results.")
def search_arxiv(query: str, max_results: int = 10) -> Optional[List[Dict[str, str]]]:
    """
    Searches arXiv for the given query using the arXiv API, then returns the search results. This is a helper function. In most cases, callers will want to use 'find_relevant_papers( query, max_results )' instead.

    Args:
        query (str): The search query.
        max_results (int, optional): The maximum number of search results to return. Defaults to 10.

    Returns:
        jresults (list): A list of dictionaries. Each dictionary contains fields such as 'title', 'authors', 'summary', and 'pdf_url'

    Example:
        >>> results = search_arxiv("attention is all you need")
        >>> print(results)
    """
    # Calculate the date four years ago
    four_years_ago = (datetime.datetime.now() - datetime.timedelta(days=4*365)).strftime('%Y-%m-%d')

    # Initialize arXiv search
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    results = []
    for result in search.results():
        # Convert publication date to string
        published_date = result.published.strftime('%Y-%m-%d')

        # Check if the paper was published in the last four years
        if published_date > four_years_ago:
            paper_info = {
                "title": result.title,
                "published": published_date,
                "authors": ', '.join(author.name for author in result.authors),
                "summary": result.summary,
                "url": result.pdf_url
            }
            results.append(paper_info)

    return results if results else None

@user_proxy.register_for_execution()
@scientific_writer.register_for_llm(description="counts the characters of the given text.")
def count_characters(output: str) -> int:
    """
    Counts and returns the number of characters in a given string.

    Args:
        output (str): The string whose characters will be counted.

    Returns:
        int: The number of characters in the input string.

    Raises:
        TypeError: If the input is not a string.
    """
    # Ensure the input is a string
    if not isinstance(output, str):
        raise TypeError("Input must be a string")
    
    # Calculate and return the character count
    character_count = len(output)
    return character_count

@user_proxy.register_for_execution()
@research_executive.register_for_llm(description="saves the given text as a .docx file")
def save_as_docx(content: str, filename: str) -> str:
    """
    Saves the given text content as a .docx file at the specified location.

    Args:
        content (str): The text content to be saved in the document.
        filename (str): The name of the file to be created
    """
    # Get the directory of the currently executing script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        # Create a new document
        doc = Document()

        # Add the content to the document
        for line in content.split('\n'):
            doc.add_paragraph(line)

        # Save the document
        file_path = os.path.join(script_directory, filename)
        doc.save(file_path)
        print(f"File {filename} saved to {file_path}")
    except Exception as e:
        print(f"Error saving document: {e}")
        return None
    
# Define group chat and manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, research_executive, scientific_researcher, literature_reader, scientific_writer, scientific_editor],
    messages=[],
    max_round=3
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Initiate chat
user_proxy.initiate_chat(
    manager,
    message="write an essay as written from a PHD-student about the usage of machine learning models in prediction of pandemics, it should have 4000 characters, save the final file",
)

research_executive.print_usage_summary()
scientific_researcher.print_usage_summary()
literature_reader.print_usage_summary()
scientific_writer.print_usage_summary()
scientific_editor.print_usage_summary()

# Gather usage summary for multiple agents
agents = [research_executive, scientific_researcher, literature_reader, scientific_writer, scientific_editor]
usage_summary = autogen.agent_utils.gather_usage_summary(agents)