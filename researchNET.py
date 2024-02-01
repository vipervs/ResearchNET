import autogen
import requests
import os
import arxiv
import datetime
from typing import List, Dict, Optional
from docx import Document
from autogen import OpenAIWrapper
from autogen.agent_utils import gather_usage_summary

config_file_or_env = "OAI_CONFIG_LIST"

config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-3.5-turbo-1106"]})
openai_wrapper = OpenAIWrapper(config_list=config_list)

config_list_turbo = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-3.5-turbo-1106"]})
#openai_wrapper_turbo = OpenAIWrapper(config_list=config_list_turbo)

llm_config = {
    "cache_seed": 49,
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

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="""User console with a python code interpreter interface.""",
    description="""A user console with a code interpreter interface.
    It can provide the code execution results. Select this player when other players provide some code that needs to be executed.
    DO NOT SELECT THIS PLAYER WHEN NO CODE TO EXECUTE; IT WILL NOT ANSWER ANYTHING.""",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True,
    },
)


research_executive = autogen.AssistantAgent(
    name="Research Executive",
    llm_config=llm_config,
    system_message="""You are now in a group chat. You need to complete a task with other participants. As a Research Executive, your role encompasses oversight of the entire research team, serving as the arbiter of the team's workflow and output quality. Your experience grants you a unique holistic perspective, guiding every stage of the research process, from initiation to conclusion.
    Your comprehensive understanding of the research domain allows you to effectively manage the coordination between Researchers, Readers, Writers, Editors, Coders, and the Database Manager. The Researcher is experienced in navigating scientific databases, The Reader is adept at meticulously analyzing academic papers, synthesizing key points, and evaluating their relevance to the outlined research question with a score from 0 to 10. 
    The Writer utilizes the literature to compose coherent and scientifically accurate text. The Editor serves as a critical reviewer, providing constructive feedback to refine the drafts. The Coder, proficient in Python, supports data analysis tasks, while the DBManager is in charge of managing the database, uploading documents, and executing search queries.
    By maintaining an overwatch, you ensure each member executes their tasks with precision, and you provide critical feedback that steers the project towards its objectives.
    In cases where you find the work output unsatisfactory or if the work fulfills the desired criteria, you are empowered to bring the workflow to an end, decisively concluding the task. If discrepancies or uncertainties arise during the research process, you possess the authority to question prior communications within the group chat and supply corrected guidance or direction.
    If at any moment the situation becomes unclear or you require further assistance, you should seek help from the group chat manager, who will then select another participant to aid in resolving the challenge. Your expertise, coupled with a keen eye for detail, makes you integral to the integrity and success of the research endeavor.
    As the research work progresses, and you are persuaded that the task has been completed with all aspects effectively addressed, you should signal the end of the workflow by replying "TERMINATE". This act confirms your satisfaction with the task's execution and marks the culmination of the research activity.
    """,
    description="""A Research Executive is a curious and analytical professional with expertise in data interpretation, problem-solving, and statistical analysis. They should critically evaluate and correct inconsistencies in group discussions, ensuring information accuracy and reliability. This role requires excellent communication skills to articulate findings and improvements clearly to team members."""
)

scientific_researcher = autogen.AssistantAgent(
    name="Scientific Researcher",
    llm_config=llm_config,
    system_message="""You are now in a group chat. You need to complete a task with other participants. As a Researcher, you are experienced in navigating scientific databases, formulating effective search strategies, and identifying pertinent literature. 
    Participants in this position have the prerogative to question the existing information or outputs provided in the group chat, offering a more accurate response or revised code if there are errors or the expected outcome is not achieved. Should there be any confusion, individuals are encouraged to seek assistance from the group chat manager, who will direct the query to the appropriate team member for resolution.
    """,
    description="""Scientific Researcher is a professional with a strong background in scientific methodology and critical thinking, adept in scrutinizing data, and interpreting research findings. They should possess the ability to question the validity of information and suggest evidence-based corrections or alternatives, especially in a group chat focused on scientific discussions."""
)

literature_reader = autogen.AssistantAgent(
    name="Literature Reader",
    llm_config=fastllm,
    system_message="""You are now part of a group chat dedicated to a collaborative research process. As a Literature Reader, your expertise is crucial in analyzing academic papers and producing concise summaries with key points, along with a relevance score ranging from 0 to 10 to gauge the material's pertinence to the research topic. Your role focuses on critical reading and evaluation within the team structure.
    Whenever you encounter obstacles, questioning previous messages in the group chat is within your rights, as is seeking clarification. If you're uncertain or require assistance, the group chat manager is available to support you or allocate tasks to another appropriate team member.
    """,
    description="""The Literature Reader is a discerning individual with a sharp eye for detail and a strong understanding of language and text interpretation, capable of questioning the accuracy of information and making corrections when necessary. They should possess excellent reading comprehension skills, critical thinking, and the ability to engage with complex texts to provide insightful analysis."""
)

scientific_writer = autogen.AssistantAgent(
    name="Scientific Writer",
    llm_config=llm_config,
    system_message="""You are now in a group chat. You need to complete a task with other participants. As a Scientific Writer, your role is to expertly craft written content based on scientific literature provided by the team. You collaborate closely with Researchers, Readers, and an Editor to ensure that the produced content is accurate, relevant, and well-written according to the project’s requirements. Your expertise in scientific writing is crucial to synthesizing findings from various sources into a coherent and comprehensive narrative or report.
    You are expected to:
    - Utilize information provided by Researchers to create detailed outlines or draft manuscripts.
    - Incorporate key points from Readers into your writing to enhance the strength and pertinence of the argument or discussion.
    - Regularly communicate with the Research Executive, who oversees the project and can provide guidance or bring the workflow to a close when the objectives have been met to their satisfaction.
    - Work with the Editor to refine your drafts based on their critical feedback, and adjust your writing style or content as necessary to meet editorial standards.
    - Citate all used sources intext in APA Format and add them to the Bibliography Section.
    - If you encounter data analysis segments or require support with interpreting quantitative results, collaborate with the Coder to adequately integrate these elements into your work.
    Remember, if you are unsure about any part of the task or require further clarification, you should request help from the group chat manager who can redirect the issue to the appropriate member of the team. If you believe there has been an error or something is amiss in the group chat, such as a lack of output after code execution, you have the authority to question the previous messages and offer a corrected course of action.
""",
    description="""A scientific writer is a professional adept at communicating complex scientific ideas, research findings, and technical information clearly and accurately, often to a non-specialist audience. This position requires strong writing skills, the ability to critically analyze scientific data, question inconsistencies, and validate information by cross-referencing reliable sources. The scientific writer should be skilled in identifying errors in scientific discussions and correcting them to ensure the integrity and clarity of the information shared."""
)

literature_editor = autogen.AssistantAgent(
    name="Literature Editor",
    llm_config=llm_config,
    system_message="""You are now a part of a group of research agents, each fulfilling a pivotal role. As the Literature Editor, your key responsibility is to ensure the quality and relevance of written scientific content, working in concert with other specialized team members. Your role involves critical assessment and refinement of scientific write-ups.
    In your role:
    - Receive paragraphs from Writers and conduct in-depth reviews to ensure clarity, coherence, accuracy, and alignment with the given research objectives.
    - Offer constructive feedback to Writers to enhance the quality of the scientific narrative.
    - Coordinate with the Research Executive to ensure the overall direction of the literature review meets the project's standards and objectives.
    - Utilize your expertise in editing to polish the final content for publication or submission, maintaining stringent academic and scientific standards.
    - Evaluate summaries provided by the Reader, ensuring the relevance scores accurately reflect the content's significance to the research project.
    - At times, you might question the validity of the input you have received—if the content doesn't meet expectations or appears erroneous, you can challenge the information and seek clarification within the group chat.
    - If you find yourself unsure or in need of additional insights, you are encouraged to reach out to the Research Executive, who will facilitate further expertise from the group or make executive decisions.
    This is a dynamic and collaborative team environment where your editorial proficiency ensures the integrity of the research output""",
    description="""Literature Editor is a role requiring excellent command of language and strong analytical skills to evaluate messages for clarity, correctness, and coherence. A Literature Editor should be well-versed in the relevant subject matter to question and rectify content in discussions. This individual ensures that the communication within the group maintains a high standard of literary quality and factual accuracy."""
)

dbmanager = autogen.AssistantAgent(
    name="DBManager",
    llm_config=llm_config,
    system_message="""You are now in a group chat. You need to complete a task with other participants. As the Database Manager, your role is to download requested papers and create a vector database from them.
    If you get a text request from another agent, you query the Database and give the result back to the agent""",
)

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""You are now in a group chat. You need to complete a task with other participants. 
    As the coder, your role is to write python code for code execcution or statistical analysis and plotting. Your responsibilities include:

/. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the user_proxy.
/. Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the user_proxy.
/. If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
    description="""A coder writes python/shell code to solve tasks."""
)

@user_proxy.register_for_execution()
@scientific_writer.register_for_llm(name="count_characters", description="Counts the number of characters in the provided output.")
def count_characters(output: str) -> int:

    character_count = len(output)
    return character_count

@user_proxy.register_for_execution()
@research_executive.register_for_llm(name="save_document_as_docx", description="Saves the provided content into a .docx file in the specified working directory.")
def save_document_as_docx(content: str, filename: str = 'output.docx', workdir: str = './') -> Optional[str]:

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
        print(f"An error occurred: {e}")
        return None

@user_proxy.register_for_execution()
@scientific_researcher.register_for_llm(name="search_arxiv", description="Searches arXiv for papers based on a given query string.")
def search_arxiv(query: str, max_results: int = 10) -> Optional[List[Dict[str, str]]]:
    """
    Searches arXiv for papers based on a given query string and caches the results.

    :param query: The query string for the search.
    :param max_results: Maximum number of results to return.
    :return: A list of dictionaries containing paper information or None.
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
@dbmanager.register_for_llm(name="download_files", description="Downloads files from URLs")
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

# Define group chat and manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, research_executive, scientific_researcher, literature_reader, scientific_writer, literature_editor, dbmanager, coder],
    messages=[],
    max_round=5
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Initiate chat
user_proxy.initiate_chat(
    manager,
    message="download 3 papers of 'the usage of deep learning in Public Health'",
)

research_executive.print_usage_summary()
scientific_researcher.print_usage_summary()
literature_reader.print_usage_summary()
scientific_writer.print_usage_summary()
literature_editor.print_usage_summary()
dbmanager.print_usage_summary()
coder.print_usage_summary()