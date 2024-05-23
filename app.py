# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

load_dotenv()

# llm=ChatGoogleGenerativeAI(
#     model = 'gemini-pro',
#     verbose=True,
#     temparature=0.6    
# )
llm = ChatOpenAI()
search_tool=DuckDuckGoSearchRun()

## Agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and data science",
    backstory="""You work at a leading tech think tank.
    Your expertise lies in idenntifying emerging trends.
    You have a knack for dissecting complex data ad presenting actionalble insights.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool]
)

writer = Agent(
    role="Tech Content Stratergist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Stratergist, known for
    your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[]    
)

task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts.
    Your final answer MUST be a fill analysis report
    """,
    agent=researcher
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
    post that highlights the most significant AI advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience.
    Make it sound cool, avoid complex words so it doesn't sound like AI.
    You final answer MUST be the full blog post of at least 4 paragraphs.
    """,
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)

result=crew.kickoff()

print("######################")
print(result)