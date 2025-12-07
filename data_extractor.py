from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile")

def extract(article):
    global llm
    prompt='''
    From the below article , extract movie name , budget , revenue , studio name in JSON output format containing
    the following keys: 'revenu_actual' , 'eps_actual' , 'revenu_expected' , 'eps_xpected'


    for each value the vaild JSON. No preamble

    Only return the valid JSON. No preamble 

    Article
    =======
    {article}

    '''

    pt=PromptTemplate.from_template(prompt)


    Chain = pt | llm
    response = Chain.invoke({article})

    parser = JsonOutputParser

    try:
        res=parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big Unable to parse job")
