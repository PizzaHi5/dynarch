#https://python.langchain.com/docs/tutorials/agents/
# import getpass
# import os
from langchain_community.tools.tavily_search import TavilySearchResults

def get_tavily_search():
    # getpass prompts user for information
    # os.environ["TAVILY_API_KEY"] = getpass.getpass()
    search = TavilySearchResults(max_results=2)
    #search_results = search.invoke("what is the weather in SF")
    #print(search_results)
    return search