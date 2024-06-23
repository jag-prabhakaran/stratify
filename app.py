from llama_index.core.agent import QueryPipelineAgentWorker
from flask import Flask, request, jsonify, render_template
from llama_index.core import SQLDatabase
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool

import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import re
import json
import os
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    AgentInputComponent,
    AgentFnComponent,
    ToolRunnerComponent
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent import Task, AgentChatResponse
from llama_index.core.llms import ChatMessage
from pyvis.network import Network
from llama_index.core.tools import BaseTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import Task, AgentChatResponse
from llama_index.core.llms import ChatResponse
import openai
from openai import OpenAI
import llama_index




app = Flask(__name__)

engine = create_engine("sqlite:///climate.db")
sql_database = SQLDatabase(engine)

os.environ['OPENAI_API_KEY'] = 'sk-proj-ZM51aF94F9s9LMnhNuH7T3BlbkFJiHyWZz423294SFitiFb2'
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_table_info(sql_database):
    metadata = MetaData()
    metadata.reflect(bind=sql_database.engine)
    tables_info = {}
    for table_name, table in metadata.tables.items():
        columns = [column.name for column in table.columns]
        tables_info[table_name] = columns
    return tables_info

tables_info = get_table_info(sql_database)

class TableInfo(BaseModel):
    table_name: str = Field(..., description="table name (must be underscores and NO spaces)")
    table_summary: str = Field(..., description="short, concise summary/caption of the table")
    column_names: List[str] = Field(..., description="list of column names in the table")

prompt_str = """\
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise.
- Do NOT output a generic table name (e.g. table, my_table).

Do NOT make the table name one of the following: {exclude_table_name_list}

Table:
{table_str}

Summary:
Column Names: {column_names}
"""

program = LLMTextCompletionProgram.from_defaults(
    output_cls=TableInfo,
    llm=LlamaOpenAI(model="gpt-4o", api_key=openai.api_key),
    prompt_template_str=prompt_str,
)

table_infos = []

for idx, (table_name, columns) in enumerate(tables_info.items()):
    table_info_path = f"{idx}_{table_name}.json"
    if os.path.exists(table_info_path):
        table_info = TableInfo.parse_file(table_info_path)
    else:
        df_str = "\n".join([f"{col}" for col in columns])
        column_names = columns
        table_info = program(
            table_str=df_str,
            exclude_table_name_list=str(list(tables_info.keys())),
            column_names=column_names,
        )
        table_info.table_name = table_name  

        with open(table_info_path, "w") as f:
            json.dump(table_info.dict(), f)

    table_infos.append(table_info)

table_info_dict = {info.table_name: info for info in table_infos}

extracted_table_names = list(tables_info.keys())
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=extracted_table_names,
    verbose=True,
)
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="sql_tool",
    description="Useful for translating a natural language query into a SQL query",
)

def agent_input_fn(task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent input function."""
    if "current_reasoning" not in state:
        state["current_reasoning"] = []
    reasoning_step = ObservationReasoningStep(observation=task.input)
    state["current_reasoning"].append(reasoning_step)
    return {"input": task.input}


agent_input_component = AgentInputComponent(fn=agent_input_fn)

def react_prompt_fn(task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]) -> List[ChatMessage]:
    chat_formatter = ReActChatFormatter()
    return chat_formatter.format(
        tools,
        chat_history=task.memory.get() + state["memory"].get_all(),
        current_reasoning=state["current_reasoning"],
    )

react_prompt_component = AgentFnComponent(fn=react_prompt_fn, partial_dict={"tools": [sql_tool]})

def parse_react_output_fn(task: Task, state: Dict[str, Any], chat_response: ChatResponse):
    """Parse ReAct output into a reasoning step."""
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}

parse_react_output = AgentFnComponent(fn=parse_react_output_fn)

def run_tool_fn(task: Task, state: Dict[str, Any], reasoning_step: ActionReasoningStep):
    """Run tool and process tool output."""
    global sql_query
    tool_runner_component = ToolRunnerComponent([sql_tool], callback_manager=task.callback_manager)
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    sql_query = ""
    if reasoning_step.action == "sql_tool":
        sql_query = reasoning_step.action_input

    print(f"SQL Query: {sql_query}")

    observation_step = ObservationReasoningStep(observation=str(tool_output))
    state["current_reasoning"].append(observation_step)

    state["sql_query"] = sql_query
    state["tool_output"] = str(tool_output)

    return {"response_str": str(tool_output), "is_done": False}

run_tool = AgentFnComponent(fn=run_tool_fn)

def process_response_fn(task: Task, state: Dict[str, Any], response_step: ResponseReasoningStep):
    """Process response."""
    state["current_reasoning"].append(response_step)
    response_str = response_step.response

    state["memory"].put(ChatMessage(content=task.input, role=MessageRole.USER))
    state["memory"].put(ChatMessage(content=response_str, role=MessageRole.ASSISTANT))

    sql_query = state.get("sql_query", "")
    tool_output = state.get("tool_output", "")
    full_response = f"SQL Query: {sql_query}\nResponse: {tool_output}"

    return {"response_str": full_response, "is_done": True}

process_response = AgentFnComponent(fn=process_response_fn)

def process_agent_response_fn(task: Task, state: Dict[str, Any], response_dict: dict):
    """Process agent response."""
    return (
        AgentChatResponse(response_dict["response_str"]),
        response_dict["is_done"],
    )


process_agent_response = AgentFnComponent(fn=process_agent_response_fn)

# Set up the Query Pipeline
qp = QP(verbose=True)

qp.add_modules(
    {
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": LlamaOpenAI(model="gpt-4o"),
        "react_output_parser": parse_react_output,
        "run_tool": run_tool,
        "process_response": process_response,
        "process_agent_response": process_agent_response,
    }
)

qp.add_chain(["agent_input", "react_prompt", "llm", "react_output_parser"])

qp.add_link(
    "react_output_parser",
    "run_tool",
    condition_fn=lambda x: not x["done"],
    input_fn=lambda x: x["reasoning_step"],
)
qp.add_link(
    "react_output_parser",
    "process_response",
    condition_fn=lambda x: x["done"],
    input_fn=lambda x: x["reasoning_step"],
)

qp.add_link("process_response", "process_agent_response")
qp.add_link("run_tool", "process_agent_response")


agent_worker = QueryPipelineAgentWorker(qp)
agent = agent_worker.as_agent(callback_manager=CallbackManager([]), verbose=True)

def create_task_with_table_context(query: str) -> Task:
    context = "\n".join([f"Table {info.table_name}: {info.table_summary} {info.table_name} columns: {info.column_names}" for info in table_infos])
    full_query = f"{query}\nHere is info about the tables: \n{context}"
    print(full_query)
    return agent.create_task(full_query)

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  

def save_python(ipt):
    py_file = open("demo.py", "w")
    py_file.write(ipt)
    py_file.close()

def execute_python_code():
    os.system("python demo.py")
    
    
def plotly_template():
    return '''figures = [
    {
        'title': 'Line Plot',
        'figure': go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='lines')),
        'insight': 'This is a line plot showing a decreasing trend.',
        'is_map': False
    },
    {
        'title': 'Bar Chart',
        'figure': go.Figure(data=go.Bar(x=[1, 2, 3], y=[2, 5, 3])),
        'insight': 'This bar chart shows that the second category has the highest value.',
        'is_map': False
    },
    {
        'title': 'Pie Chart',
        'figure': go.Figure(data=go.Pie(labels=['A', 'B', 'C'], values=[30, 50, 20])),
        'insight': 'This pie chart shows that category B is the largest.',
        'is_map': False
    },
    {
        'title': 'Scatter Plot',
        'figure': go.Figure(data=go.Scatter(x=[1, 2, 3], y=[2, 4, 5], mode='markers')),
        'insight': 'This scatter plot shows a positive correlation.',
        'is_map': False
    },
    {
        'title': 'Map Plot',
        'figure': go.Figure(data=go.Scattergeo(lon=[-75, -80, -70], lat=[45, 50, 40], mode='markers')).update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0}),
        'insight': 'This is a map plot showing geographic locations.',
        'is_map': True
    }
    ]'''
    
def context(query: str) -> Task:
    context = "\n".join([f"Table {info.table_name}: {info.table_summary} {info.table_name} columns: {info.column_names}" for info in table_infos])
    full_query = f"{query}\nHere is info about the tables: \n{context}"
    return full_query
    
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
    adhoc = False
    if adhoc:
        task = create_task_with_table_context(user_query)
        step_output = agent.run_step(task.task_id)
        step_output.is_last = True
        response = agent.finalize_response(task.task_id)
        result = sql_query['input']
        database = "climate"
        system_role = f'Write python code to select relevant data. Use the SQL query provided to connect to the Database and retrieve data. The database name is {database}.db. Please create a data frame from relevant data and print the dataframe. Only use the sql i provide and do not generate your own. create a function called return_df() to return the df. The db name is {database}. Do not limit data at all.'
        max_tokens = 2500
        
        question = f"Question: {user_query} \nSQL Query: {result}"
        
        client = OpenAI()

        def get_gpt_result(system_role, question, max_tokens):
            response = client.chat.completions.create(
                model="gpt-4",
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": question}
                ]
            )

            return response
        
        response = get_gpt_result(system_role, question, max_tokens)
        text = response.choices[0].message.content
        try:
            matches = find_all(text, "```")
            matches_list = [x for x in matches]

            python = text[matches_list[0] + 10:matches_list[1]]
        except:
            python = text
        
        save_python(python)
        from demo import return_df
        df = return_df()
        data = df.to_string()
        question = "Question: " + user_query + '\nData: \n' + data + '\nOnline information: \n'
        print(f'QUESTION: {question}')
        system_role2 = 'Generate analysis and insights about the data in 5 bullet points.  Only choose relevant information to generate deeper insights. Use actual numbers in each bulletpoint to validate your analysis. do not simply just use words. Do not ever say data provided does not include information. or anything along the lines of data is limiting. every query is valid and in the database.'
        response2 = get_gpt_result(system_role2, question, max_tokens)
        text2 = response2.choices[0].message.content
        
        return jsonify({"sql_query": text2})
    else:
        max_tokens = 2500
        print("starting else:")
        system_role = '''generate questions to serve as basis to generate sql query. you simply return 5 bullet point questions do not write any queries.'''
        quest_order = 'can you give me 5 question to track co2 emissions per country, change in temperature over time, types of energy consumptions, and to your choice 2 different interesting graphs from the data provided. ythese quesitons serve as the basis for me to generate sql queries. There should be 5 types of graphs that plotly can make. these questions should answer for 1. map graph, 2. scatter plot, 3 bar graph, 4. pie chart, 5. line graph. I need quesitons to generate exactly those 5 graphs. Do not give vague questions. say with specificity. i.e no saying specific year give me the actual year instead. okay so also dont tell me what type of graph everything is. just follow this order of 1-5 and those types as well.'
        
        database = "climate"
        cont = context(quest_order)
        question = quest_order + "\n" + cont
        
        client = OpenAI()

        def get_gpt_result(system_role, question, max_tokens):
            response = client.chat.completions.create(
                model="gpt-4",
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": question}
                ]
            )

            return response
        response = get_gpt_result(system_role, question, max_tokens)
        text = response.choices[0].message.content
        question_list = text.split("\n")
        print(f'these are the questions beins asked {question_list}')
        
        sql_queries = []
        for question in question_list:
            task = create_task_with_table_context(question)
            step_output = agent.run_step(task.task_id)
            step_output.is_last = True
            response = agent.finalize_response(task.task_id)
            result = sql_query['input']
            sql_queries.append(result)
        dfs_list = []
        for q in sql_queries:
            system_role = f'Write python code to select relevant data. Use the SQL query provided to connect to the Database and retrieve data. The database name is {database}.db. Please create a data frame from relevant data and print the dataframe. Only use the sql i provide and do not generate your own. create a function called return_df() to return the df. The db name is {database}. Do not limit data at all.'
            max_tokens = 2500
            
            question = f"Question: {user_query} \nSQL Query: {result}"
            
            client = OpenAI()
            
            response = get_gpt_result(system_role, question, max_tokens)
            text = response.choices[0].message.content
            try:
                matches = find_all(text, "```")
                matches_list = [x for x in matches]

                python = text[matches_list[0] + 10:matches_list[1]]
            except:
                python = text
            
            save_python(python)
            from demo import return_df
            df = return_df()
            data = df.to_string()
            dfs_list.append(data)
        
        temp = plotly_template()
        system_role = f'Generate python code for plotly graphs in the provided template for the requested infromation along with the questions they are answering for context with the df in mind. the plots should be in this order: 1. map graph, 2. scatter plot, 3 bar graph, 4. pie chart, 5. line graph. Only use this format and return in square brackets as such do not return anything else.\n' + temp + "Questions for context:\n" + ",".join(question_list)
        question = ''
        for i in range(len(dfs_list)):
            question = "1: " + dfs_list[i]
        
        response = get_gpt_result(system_role, question, max_tokens)
        text = response.choices[0].message.content
        
        return jsonify({"sql_query": text})
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8222)