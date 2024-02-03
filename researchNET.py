import autogen
import requests
import os
import arxiv
import datetime
from typing import List, Dict, Optional
from docx import Document
from autogen import OpenAIWrapper
from langchain.tools import PubmedQueryRun

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
    system_message="""You are the research_executive in a group chat focused on completing a research task. Your role is to oversee the research team, ensuring quality and workflow efficiency. Your key responsibilities include:
    - Collaborate with scientific_researcher for paper gathering, literature_reader for paper analysis, scientific_writer for drafting text, scientific_editor for content refinement, and coder for data analysis support. 
    - formulate a topic and outline best suited for the task and delegate it to the scientific_researcher.
    - If the Group Chat Manager gives another task guide the team to solve this task.
    - Evaluate the relevance and quality of work, providing feedback for improvement.
    - In case of discrepancies or questions, direct the team for corrections or seek clarification.
    - Signal "TERMINATE" to mark the completion of the task, ensuring all aspects are effectively addressed.""",
    description="""A research_executive coordinates the research team, guiding task delegation, ensuring quality, providing feedback, and signaling task completion to achieve research objectives efficiently."""
)

scientific_researcher = autogen.AssistantAgent(
    name="scientific_researcher",
    llm_config=llm_config,
    system_message="""As a scientific_researcher in a group chat, your expertise lies in exploring scientific databases and identifying relevant literature. Your key responsibilities include:
    - Formulate relevant keywords to the provided topic and outline.
    - Use your ability to search in scientific databases for proper papers.
    - Provide the found literature without changing the output to the literature_reader.
    - The literature_reader can give you orders to find additional literature if he is not satisfied with your results.
    If uncertainties arise or further assistance is needed, reach out to the research_executive for guidance or redirection.""",
    description="""A scientific_researcher specializes in efficiently sourcing and providing relevant scientific literature from databases like PubMed and arXiv, tailored to the research team's needs, ready for further analysis by the Literature Reader."""
)

literature_reader = autogen.AssistantAgent(
    name="literature_reader",
    llm_config=llm_config,
    system_message="""As a literature_reader in a group chat, your primary task is to analyze academic papers, distill key points, and assign a relevance score from 0 to 10, reflecting each material's importance to the research topic. Your key responsibilities include:
    - Critically evaluate academic literature found by the literature_reader and summarize key findings.
    - Rate the relevance of each piece of literature to the provided topic and outline.
    - Request aditional literature from the scientific_researcher if you are not satisfied with the provided literature.
    - If you are satisfied with the literature and think the writing task for the given topic and outline can be completed with it, delegate it to the Scientifc Writer.
    If uncertainties arise or further assistance is needed, reach out to the research_executivee for guidance or redirection.""",
    description="""A literature_reader critically evaluates and summarizes academic papers sourced by the scientific_researcher, assigning relevance scores to guide further research analysis and requesting additional literature as needed."""
)

scientific_writer = autogen.AssistantAgent(
    name="scientific_writer",
    llm_config=llm_config,
    system_message="""As a scientific_writer in a group chat, you are tasked with producing written content that integrates findings from scientific literature. Your key responsibilities include:
    - Drafting a manuscript for the given topic and outline using literature provided by the literature_reader.
    - Writing in  a concise an scientific writing style.
    - Counting the characters to reach the goal provided by the research_executive.
    - If you cant reach the needed characters, request more literature from the scientific_researcher.
    - Cite the used literature in APA format and organize them into a Bibliography Section.
    - Engange with the Coder if you need to include data analysis into the narrative.
    - When your draft is completed, give it to the scientific_editor.
    - Refine drafts in collaboration with the Editor based on feedback.
    If uncertainties arise or corrections are needed, consult the research_executive for support or to address discrepancies within the chat.""",
    description="""A scientific_writer crafts comprehensive manuscripts based on provided literature, ensuring character count goals are met with proper citation, and collaborates with Editors and Coders for refinement and data integration."""
)

scientific_editor = autogen.AssistantAgent(
    name="scientific_editor",
    llm_config=llm_config,
    system_message="""As the scientific_editor in a group chat, your primary duty is to refine and validate the content produced by the Scientifc Writer. Your key responsibilities include:
    - Review the content from the scientific_writer for clarity, coherence, and scientific accuracy, ensuring it aligns with topic and outline and is proper cited according to the literature and bibliography section.
    - If you are not satisfied provide detailed feedback for the scientific_writer to improve the scientific narrative's quality.
    - If the manuscript is ready for final submissions give it to the research_executive.
    If uncertainties arise or corrections are needed, consult the research_executive for support or to address discrepancies within the chat.""",
    description="""A scientific_editor critically reviews and refines manuscripts from the scientific_writer for clarity, coherence, and accuracy, providing feedback for improvements and preparing content for final evaluation by the Research Executive."""
)

coder = autogen.AssistantAgent(
    name="coder",
    llm_config=llm_config,
    system_message="""As a Coder in a group chat, your primary task involves writing Python or shell scripts for code execution, statistical analysis, and data visualization. Your key responsibilities include:
    - Ensure all code is complete and executable as provided; the user cannot modify your submissions. Always wrap your code in a code block.
    - Submit one code block per response to avoid confusion and ensure clarity in execution.
    - Directly check execution results from the user_proxy. If errors occur, revise and resubmit the corrected code.
    - Avoid partial code or incremental updates. If an approach fails, reassess your strategy, gather any additional information needed, and propose a new, full solution.
    Your coding expertise is crucial for solving tasks efficiently and supporting the team's research objectives.""",
    description="""A coder specializes in developing Python/shell scripts for task resolution, playing a key role in data analysis and visualization within research projects."""
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
def save_as_docx(content: str, filename: str = 'output.docx', workdir: str = './') -> Optional[str]:
    """
    Saves the given text content as a .docx file at the specified location.

    Args:
        content (str): The text content to be saved in the document.
        filename (str, optional): The name of the file to be created. Defaults to 'output.docx'.
        workdir (str, optional): The directory where the file will be saved. Defaults to './'.

    Returns:
        Optional[str]: The path to the saved document if successful, None otherwise.
    """
    try:
        # Ensure the working directory exists
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        # Create a new document
        doc = Document()

        # Add the content to the document
        for line in content.split('\n'):
            doc.add_paragraph(line)

        # Save the document
        file_path = os.path.join(workdir, filename)
        doc.save(file_path)
        return file_path
    except Exception as e:
        return None
    
# Define group chat and manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, research_executive, scientific_researcher, literature_reader, scientific_writer, scientific_editor, coder],
    messages=[],
    max_round=5
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Initiate chat
user_proxy.initiate_chat(
    manager,
    message="write an abstract about leveraging digital public health with the help of AI and ML with the following requirements: - search for literature not older than 1 year to write it - it should have the following sections; Objective, Methods, Results, Conclusions - the final abstract should have a minimum of 1500 characters - on the bottom cite the literature you sued to write it - save the final abstract as a .docx file",
)

research_executive.print_usage_summary()
scientific_researcher.print_usage_summary()
literature_reader.print_usage_summary()
scientific_writer.print_usage_summary()
scientific_editor.print_usage_summary()
coder.print_usage_summary()