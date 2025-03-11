from time import time
from datetime import datetime

print(f"Starting, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import pandas as pd
from langchain_openai import OpenAI
import ast
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults

print(f"Starting, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

from llama_index.core.workflow import Event
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)

print(f"Starting, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import pandas as pd
import asyncio


df = pd.read_csv("menudata.csv")

embeddings = OpenAIEmbeddings()

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True, allow_dangerous_code=True,)

pandas_agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4o"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    return_intermediate_steps=True
)

ROUTER_PROMPT = """
You are an expert in identifying what is the plan of action to answer the question.
our database has the following columns:
restaurant_name: 
menu_category: Category of the menu item - eg: Appetizers, Main Course, Desserts, Drinks, champagne, chefs tasting, available 11-9pm, etc.
menu_item: 
menu_description:
ingredient_name: 
categories: Categories of the menu item
address1:
city: City of the restaurant - always San Francisco
confidence_score: Confidence score of the ingredient
rating: Rating of the restaurant, duplicated per menu item
zip_code: Zip code of the restaurant - Always 94110
state: State of the restaurant - always CA
country: Country of the restaurant - always US
rating: Rating of the restaurant
review_count: Number of reviews of the restaurant, duplicated per menu item
price: denoted by $ to $$$$

So, if a question had a different city, or zip_code, then we have to use tavily search tool.
The individual options are:
1. Use the pandas agent to answer the question using the dataframe.
2. Use the tavily search agent to answer the question.

Return the option *Always* as a list of lists. the question could be split if more than 1 option is chosen.
Reasons in examples are present just to give you an idea. Do not return anything apart from the list of lists.

Question example: Compare the average menu price of vegan restaurants in San Francisco vs. in New Mexico.
[["pandas", "average menu price of vegan restaurants in San Francisco"], ["tavily", "average menu price of Vegan restaurants in New Mexico"]]
Reason: No mention of any restaurants in New Mexico in our database, so have to rely on tavily search for that.

Question example: What is the history of sushi, and which restaurants in my area are known for it?
[["pandas", "Best restaurants known for sushi in my area"], ["tavily", "history of sushi"]]
Reason: History, obviously cannot be answered using the dataframe. but restaurant options can be answered using the dataframe.

Question example: How has the use of saffron in desserts changed over the last year, according to restaurant menus or news articles?
[["tavily", "How has the use of saffron in desserts changed over the last year, according to restaurant menus or news articles?"]]
Reason: Time sensitive information is not present in the database so we have to rely on tavily search for that.

Question: {query}
"""

ROUTER_LLM = ChatOpenAI(
    model='gpt-4o',
    temperature=0.1,
    max_tokens=2048
)

router_prompt_template = PromptTemplate(template=ROUTER_PROMPT, input_variables=["query"])
router_chain = router_prompt_template | ROUTER_LLM

COLUMN_DECIDER_PROMPT = """
Human: You are an expert in identifying which columns are relevant to the question.
The purpose of this is to get all the relevant column names with the relevant snippet to retrieve from that column.
Here are columns and its description:

restaurant_name: Name of the restaurant
menu_category: Category of the menu item - eg: Appetizers, Main Course, Desserts, Drinks, champagne, chefs tasting, available 11-9pm, etc.
menu_item: Name of the menu item
menu_description: Description of the menu item
ingredient_name: Name of the ingredient in the menu item
categories: Categories of the menu item
address1: Address of the restaurant
city: City of the restaurant always San Francisco
price: Price of the menu item denoted by $ to $$$$
confidence_score: Confidence score of the ingredient - 0 to 1
rating: Rating of the restaurant, duplicated per menu item
review_count: Number of reviews of the restaurant, duplicated per menu item
zip_code: Zip code of the restaurant - Always 94110
state: State of the restaurant - always CA
country: Country of the restaurant - always US


if the category is unsure, then send all possible relevant columns it could be from. 
the output *strictly* should be in a list of lists format:

Eg:

1. Question: Name all restaurants in courtland ave that might serve sushi
[["address1", "courtland ave"],["menu_category", "sushi"],["menu_item", "sushi"],["menu_description", "sushi"],["ingredient_name", "sushi"]["categories", "sushi"]]

2. Question: All restaurants in San Francisco that serve impossible meat
[["city", "San Francisco"],["menu_category", "impossible meat"],["menu_item", "impossible meat"],["menu_description", "impossible meat"],["ingredient_name", "impossible meat"]["categories", "impossible meat"]]


Actual Question: {query}
"""

COLUMN_DECIDER_LLM = ChatOpenAI(
    model='gpt-4o',
    temperature=0.1,
    max_tokens=2048
)

column_decider_prompt_template = PromptTemplate(template=COLUMN_DECIDER_PROMPT, input_variables=["query"])
column_decider_chain = column_decider_prompt_template | COLUMN_DECIDER_LLM


FINAL_ANSWER_PROMPT = """
Human: You are an expert in combining the results from the database and the tavily search and answering the question appropriately.

Here is the question:
{query}

Here are the results from the database:
{database_results}

Here are the results from the tavily search:
{tavily_results}

Answer the question based on the above results. Do not mention database or tavily. 
Respond like you are saing it from your own knowledge. 
You can say database info like ratings and stuff but not phrasing like "the database says" or "tavily says" or anything like that.
if you cannot answer the question based on the above results, then say <<NO_ANSWER>>.
"""

final_answer_llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.1,
    max_tokens=2048
)

final_answer_prompt_template = PromptTemplate(template=FINAL_ANSWER_PROMPT, input_variables=["query", "database_results", "tavily_results"])
final_answer_chain = final_answer_prompt_template | final_answer_llm

tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
)

points_to_remember = """
Some additional points to remember:
1. if the question is about vegan, do not consider veegetarian or veggie as keywords
2. if questions about price, if needed, scale $ by 10. so $$ = 20, $$$ = 30, $$$$ = 40...
3. Try returning the dataframe whenever possible.
4. the rows are because of ingredients. So if questions like top 2 restaurants are asked, then there has to be some transformation where you find uniques and process that.
"""

class RouterEvent(Event):
    """Router event (routes to the correct workflow)."""
    pass

class RetrieveEvent(Event):
    """Retrieve event (gets retrieved results)."""

    query: str
    
class TavilySearchEvent(Event):
    """Tavily search event (searches tavily)."""

    query: str

class PandasAgentEvent(Event):
    """Pandas agent event (uses a pandas agent)."""

    initial_split_query: str
    string: str

class PandasCompleteEvent(Event):
    """Pandas complete event (uses a pandas agent)."""

    query: str
    results: dict
    event: str
class TavilyCompleteEvent(Event):
    """Tavily complete event (searches tavily)."""

    query: str
    results: dict
    event: str
class CombinerEvent(Event):
    """Combiner event (combines the results)."""

    query: str
    results: list[str]

class final_answer_event(Event):
    """Final answer event - after getting all the results (final answer)."""

    query: str
    results: dict

class MenuDataWorkflow(Workflow):
    """Menu data workflow."""

    async def get_retrievals(self, parsed_list):
        # load vector store
        menudata_vector_store = FAISS.load_local(
            "menudata", embeddings, allow_dangerous_deserialization=True
        )
        # List to store all retrievals
        all_retrievals = []
        
        # Process each [column_name, value] pair asynchronously
        async def process_pair(pair):
            column_name, search_value = pair
            try:   
                # Create a wrapper function that includes all arguments
                def search_wrapper():
                    return menudata_vector_store.similarity_search_with_score(
                        search_value,
                        k=2,
                        filter={"column": column_name}
                    )
                
                # Run the wrapper function in the executor
                retrieval = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    search_wrapper
                )

                # Format the results
                result_strings = []
                for doc, score in retrieval:
                    result_strings.append(f"Content: {doc.page_content}\nScore: {score}")
                full_string = "\n\n".join(result_strings)
                
                return {"column": column_name, "results": full_string}
            except Exception as e:
                print(f"Error retrieving for {column_name}: {search_value} - {str(e)}")
                return None

        # Create tasks for all pairs
        tasks = [process_pair(pair) for pair in parsed_list]
        
        # Gather all results
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and add to retrievals
        all_retrievals.extend([r for r in results if r is not None])
        # for r in all_retrievals:
        #     # print(f'{r}\n\n')
        return all_retrievals
    

    @step
    async def router(self, ctx: Context, ev: StartEvent) -> RetrieveEvent | TavilySearchEvent | StopEvent:
        """Router step."""
        query: str | None = ev.get("query")
        print(f'query: {query}')
        plan_of_action: list[list[str]] | None = ev.get("plan_of_action", default=[])
        print(f'plan of action: {plan_of_action}')

        await ctx.set("intermediate_results", {})
        await ctx.set("query", query)
        await ctx.set("pandas_count", 0)
        await ctx.set("tavily_count", 0)
        
        answer = router_chain.invoke({
            "query": query
            })
        
        print(f'answer: {answer}')
        plan_of_action = ast.literal_eval(answer.content)

        # testing

        # plan_of_action = [["tavily", "History of Pasta"]]
        
        # print(f'plan of action: {plan_of_action}')
        await ctx.set("plan_of_action", plan_of_action)
        action_to_event_map = {
            "pandas": RetrieveEvent,
            "tavily": TavilySearchEvent
        }

        # Send events for each action in the plan
        for action, query_text in plan_of_action:
            action = action.lower()
            if action in action_to_event_map:
                await ctx.set(action, action)
                await ctx.set(f"{action}_count", await ctx.get(f"{action}_count", 0) + 1)
                event_class = action_to_event_map[action]
                ctx.send_event(event_class(query=query_text))
            else:
                print(f"Warning: Unknown action type {action}")

        # return StopEvent(result="Router complete")
    
    @step(num_workers = 2)
    async def retrieve(self, ctx: Context, ev: RetrieveEvent) -> PandasAgentEvent | StopEvent:
        """Retrieve step."""
        query = ev.query
        # print(f'query in retrieve: {query}')
        column_decider_chain_answer = column_decider_chain.invoke({
                                        "query":query
                                        })
        
        # print(f'column decider chain answer: {column_decider_chain_answer.content}')

        columns_to_retrieve = ast.literal_eval(column_decider_chain_answer.content)
        
        # list of dicts
        column_content_list = await self.get_retrievals(columns_to_retrieve)
        full_string = "Here are the columns and to most similar results from the database:\n"
        for entry in column_content_list:
            full_string += f'{entry["column"]}:\n{entry["results"]}\n\n'
        full_string += '\n There are other columns too so include all the columns in the final df answer.'
        # print(f'full string: {full_string}')
        return PandasAgentEvent(initial_split_query=query, string=full_string)
        
    @step(num_workers = 2)
    async def pandas_agent(self, ctx: Context, ev: PandasAgentEvent) -> PandasCompleteEvent | StopEvent:
        """Pandas agent step."""
        pandas_query = ev.initial_split_query
        pandas_string = ev.string + points_to_remember
        full_query = f"{pandas_query}\n\n{pandas_string}"
        print(f'full query to send to pandas agent:\n {full_query}\n\n\n')
        pandas_agent_answer = pandas_agent.invoke(full_query)
        print(f'pandas agent answer:\n {pandas_agent_answer}\n\n\n')

        # handle potential None errors, saw in one of the runs.
        try:

            df = pandas_agent_answer["intermediate_steps"][-1][1]
            if isinstance(df, pd.DataFrame):
                print("df is a dataframe")
            elif isinstance(df, tuple):
                df = df[0]
            else:
                print("df is not a dataframe or tuple")
                # print(f"\n\n\n\n\n{df}\n\n\n\n\n")
            print(f'\n\n\n\n\ndf: {df}\n\n\n\n\n{type(df)}\n\n\n\n\n')
        except:
            print("df not found")
            print(f"\n\n\n\n\n{pandas_agent_answer}\n\n\n\n\n")
            df = None
        try:
            item_ids = df['item_id'].tolist()
        except:
            item_ids = []

        return PandasCompleteEvent(
                        event = "pandas",
                        query=full_query, 
                        results={"answer": pandas_agent_answer["output"], "df": df, "item_ids": item_ids}
                        )

    @step(num_workers = 2)
    async def tavily_search(self, ctx: Context, ev: TavilySearchEvent) -> TavilyCompleteEvent | StopEvent:
        """Tavily search step."""
        query = ev.query
        # print(f'query in tavily search: {query}')
        num_retrievals = await ctx.get("num_retrievals", 5)
        if num_retrievals > 5:
            print(f'num retrievals: {num_retrievals}')
            tavily_tool_2 = TavilySearchResults(
                max_results=num_retrievals,
                search_depth="advanced",
                include_answer=True,
            )
            answer = tavily_tool_2.invoke({"query": query})
        else:
            pass
            answer = tavily_tool.invoke({"query": query})
        content = "Content:\n" + '\n\n'.join([f"{idx+1}. {item['content']}" for idx, item in enumerate(answer)])
        references = [item['url'] for item in answer]
        print(content)
        print(references)
        # content = 'hi, this is a test response'
        # references = ['123.txt', '456.txt']
        # await asyncio.sleep(5)
        return TavilyCompleteEvent(
                        event = "tavily",
                        query=query, 
                        results={"content": content, "references": references}
                        )

    @step
    async def combiner(self, ctx: Context, ev: PandasCompleteEvent | TavilyCompleteEvent) -> final_answer_event:
        """Combiner step."""
        # print("Received event ", ev.results)
        pandas_count = await ctx.get("pandas_count", 0)
        tavily_count = await ctx.get("tavily_count", 0)
        intermediate_results = await ctx.get("intermediate_results", default={})
        if ev.event == "pandas":
            if intermediate_results.get("pandas_results") is None:
                intermediate_results["pandas_results"] = [ev.results]
            else:
                intermediate_results["pandas_results"].append(ev.results)
        elif ev.event == "tavily":
            if intermediate_results.get("tavily_results") is None:
                intermediate_results["tavily_results"] = [ev.results]
            else:
                intermediate_results["tavily_results"].append(ev.results)
        await ctx.set("intermediate_results", intermediate_results)

        # Create arrays of events based on counts
        events = []
        if pandas_count > 0:
            events.extend([PandasCompleteEvent] * pandas_count)
        if tavily_count > 0:
            events.extend([TavilyCompleteEvent] * tavily_count)

        print(f'events to collect: {str(events)}')

        results = ctx.collect_events(ev, events)

        if results is None:
            return None

        intermediate_results = await ctx.get('intermediate_results')

        return final_answer_event(
            query= await ctx.get("query"), 
            results={
                "database_results": intermediate_results.get("pandas_results", []), 
                "tavily_results": intermediate_results.get("tavily_results", [])
                })

    @step
    async def final_answer(self, ctx: Context, ev: final_answer_event) -> StopEvent:
        """Final answer step."""
        initial_query = await ctx.get("query")
        results = ev.results
        # print("Received event ", results)
        db_content = ''
        tavily_content = ''
        for db in results["database_results"]:
            db_content += f'{db["answer"]}\n\n'
        for tav in results["tavily_results"]:
            tavily_content += f'{tav["content"]}\n\n'
        final_answer = final_answer_chain.invoke({
            "query": initial_query,
            "database_results": db_content,
            "tavily_results": tavily_content
        })

        print(f'final answer: {final_answer.content}')

        if final_answer.content == "<<NO_ANSWER>>":
            if await ctx.get("retry_count", 0) > 0:
                return StopEvent(result={
                                "final_answer": "I was unable to answer the question. Please try again, maybe with more context", 
                                "db_results": [], 
                                "tavily_results": []
                            })
            else:
                await ctx.set("tavily_count", 1)
                await ctx.set("retry_count", await ctx.get("retry_count", 0) + 1)
                await ctx.set("pandas_count", 0)
                await ctx.set("plan_of_action", [["tavily", initial_query]])
                await ctx.set("num_retrievals", 10)

                print(f'''
                      variables set to retry: 
                      Tavily count: {await ctx.get("tavily_count")}, 
                      Retry count: {await ctx.get("retry_count")}, 
                      Pandas count: {await ctx.get("pandas_count")}, 
                      Plan of action: {await ctx.get("plan_of_action")}
                      ''')
                
                return TavilySearchEvent(query=initial_query)
            
        else:
            return StopEvent(result={
                                "final_answer": final_answer.content, 
                                "db_results": results["database_results"], 
                                "tavily_results": results["tavily_results"]
                            })

async def execute_loop(query):

    wf = MenuDataWorkflow(timeout=60.0)
    result = await wf.run(query=query)
    return result

async def main(query):
    result = await execute_loop(query=query)
    print(f'\n\n\n\n\n\nresult: {result}\n\n\n\n\n\n')
    final_output = result['final_answer']
    tavily_results = result.get('tavily_results',[])
    db_results = result.get('db_results',[])
    tavily_references = []
    db_item_ids = []
    for tav in tavily_results:
        tavily_references += [ref for ref in tav.get('references',[])]
    for db in db_results:
        db_item_ids += [item_id for item_id in db.get('item_ids',[])]

    return final_output, tavily_references, db_item_ids


if __name__ == "__main__":
    # q1: What is the history of sushi, and which one restaurant in my area is known for it?
    # q2: Find restaurants near me that serve gluten-free pizza
    # q3: Compare the average menu price of vegan restaurants in San Francisco vs. Mexican restaurants
    # q4: How has the use of saffron in desserts changed over the last year, according to restaurant menus or news articles?
    answer, tavily_references, db_item_ids = asyncio.run(main("How has the use of saffron in desserts changed over the last year, according to restaurant menus or news articles?"))
    print(answer)
    print(tavily_references)
    print(db_item_ids)



    