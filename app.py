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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
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
    system_role2 = 'Generate analysis and insights about the data in 5 bullet points. Online information is provided but may not be relevant to the question. Only choose relevant information to generate deeper insights.'
    response2 = get_gpt_result(system_role2, question, max_tokens)
    text2 = response2.choices[0].message.content
    
    return jsonify({"sql_query": text2})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8222)
