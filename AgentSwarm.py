import autogen
from autogen.agentchat.contrib.agent_builder import AgentBuilder

# 1. Configuration
config_path = 'OAI_CONFIG_LIST.json'
config_list = autogen.config_list_from_json(config_path)
default_llm_config = {'temperature': 0}

# 2. Initialising Builder
# builder = AgentBuilder(config_path=config_path)
builder = AgentBuilder(config_file_or_env=config_path)


# 3. Building agents
building_task = "Find a paper on arxiv by programming, and analyze its application in some domain..."
agent_list, agent_configs = builder.build(building_task, default_llm_config)



# 4. Multi-agent group chat
group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **default_llm_config})
agent_list[0].initiate_chat(
    manager, 
    message="Find a recent paper about gpt-4 on arxiv..."
)