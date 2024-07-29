import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from dotenv import load_dotenv
import requests, os
import json
from openai import AzureOpenAI, BadRequestError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import urllib.request
import ssl
import winsound


load_dotenv()

# client = openai.OpenAI()

config_list = [
    {
        'model': os.getenv('AZURE_OPENAI_MODEL'),
        'api_key': os.getenv('AZURE_OPENAI_KEY'),
        'base_url': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'api_type': 'azure',
        'api_version': os.getenv('AZURE_OPENAI_VERSION')
    }
]

#-------------------Get Customer Information-------------------
def get_customer_information(phonenumber):

    # return customer name, address, and account number and phone number in json format
    customer_info = {
        "name": "John Doe",
        "address": "123 Main St",
        "account_number": "0000-9999-8888",
        "phone_number": phonenumber
    }
    return json.dumps(customer_info)

#-------------------Get Promotions using account number-------------------
def get_promotions(account_number):
    promotions = {
        "free_hulu_service": True,
        "discount_for_additional_line": "$10",
        "internet_speed_upgrade": "50 Mbps"
    }
    return json.dumps(promotions)

    

#-------------------RAG Search InfoBot Knowledge Base-------------------
#**Important** This is calling an Endpoint in Azure ML Studio which in turn calls the RAG model
#You may need to replace the URL with the correct URL for your deployment
def get_answer_from_kb(question):
    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    data = {"question": question}

    body = str.encode(json.dumps(data))

    url = '' # Replace this with the endpoint URL for the deployment
    # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
    api_key = ''
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'vz-infobot-oai-chat-1' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        return error.info()


#-------------------Get Usage Information for the customer-------------------
def get_usage_by_customer(account_number):
    usage_history = {
        "123-456-7890": [ 
            {"2024": 
             [
                {"Month": "June", "minutes": 150, "bill_amount": 172.50},
                {"Month": "May", "minutes": 160, "bill_amount": 182.50},
                {"Month": "April", "minutes": 170, "bill_amount": 192.50},
                {"Month": "March", "minutes": 140, "bill_amount": 162.50},
                {"Month": "February", "minutes": 155, "bill_amount": 179.75},
                {"Month": "January", "minutes": 165, "bill_amount": 183.75}
            ]
            },
        
        ],
        "972-765-4321": [
            {"2024": 
             [
                {"Month": "June", "minutes": 90, "bill_amount": 55.50},
                {"Month": "May", "minutes": 80, "bill_amount": 47.50},
                {"Month": "April", "minutes": 79, "bill_amount": 47.00},
                {"Month": "March", "minutes": 111, "bill_amount": 79.50},
                {"Month": "February", "minutes": 121, "bill_amount": 99.75},
                {"Month": "January", "minutes": 83, "bill_amount": 50.75}
            ]
            },
        
        ],


    }

    return json.dumps(usage_history)



#-------------------Main Program-------------------
# Define the tools list with the functions and their descriptions
tools_list = [
{
    "type": "function",
    "function": {
        "name": "get_customer_information",
        "description": "Get the customer information based on their phone number",
        "parameters": {
            "type": "object",
            "properties": {
                "phonenumber": {
                    "type": "string",
                    "description": "Customer phone number",
                },
            },
            "required": ["phonenumber"]
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "get_promotions",
        "description": "Get the sales promotions for the customer based on their account number",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {
                    "type": "string",
                    "description": "Customer account number",
                },
            },
            "required": ["account_number"]
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "get_answer_from_kb",
        "description": "Answer customer query using the data from knowledge base",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "customer query to be answered using the knowledge base",
                },
            },
            "required": ["question"]
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "get_usage_by_customer",
        "description": "Get the customer phone(s) usage information by Year, Month, minutes and bill amount",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {
                    "type": "string",
                    "description": "Customer account number",
                },
            },
            "required": ["account_number"]
        },
    },
}
]


llm_config = { 
    "tools": tools_list,
    "config_list": config_list,
    "temperature": 0
}

# Create the GPT Assistant Agent which is responsible for creating the function calls, and generating the code
gpt_assistant = GPTAssistantAgent(
    name="Developer Assistant",
    instructions="""
    Developer Assistant who performs tasks
    Reply TERMINATE when the task is solved and there is no problem
    """,
    llm_config=llm_config
)

# Register the functions with the GPT Assistant
gpt_assistant.register_function(
    function_map={
        "get_customer_information": get_customer_information,
        "get_promotions": get_promotions,
        "get_answer_from_kb": get_answer_from_kb,
        "get_usage_by_customer": get_usage_by_customer
    }

)

# Create the User Proxy Agent which is responsible for interacting with the user
#This also execures the functions and generated code
#In production scenarios use docker for code execution
user_proxy = autogen.UserProxyAgent(
    name="Call Center Agent",
    human_input_mode="NEVER", # values are "TERMINATE", "ALWAYS", "NEVER"
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=3,
    code_execution_config={
        "work_dir" : "code",
        "use_docker": False,
    }
)


while True:
    # Get user input and display text in green color
    winsound.Beep(1000, 500)
    user_input = input("\033[92mEnter user question: \033[0m")

    if user_input == '':
        user_input = """Customer information for phone number 123-456-7890,
        promotions available, 
        generate python code and create graph of customer usage for each phone number by year and month. Save it in the code folder with the 'usage.png' filename and display it,
        Do customer need to qualify an address for 5G service?"""

    if user_input == 'q':
        break

    chat_result = user_proxy.initiate_chat(
            gpt_assistant,
            message=user_input
            )
    # print(chat_result.chat_history)
    # print(chat_result.cost) 
    # print(chat_result.summary)

print("Goodbye!")