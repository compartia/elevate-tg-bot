# from __future__ import annotations
#
# import logging
# from pathlib import Path
#
# import httpx
# import openai
# from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic
#
# from persistence import JSONFileConversationPersistence
#
#
# def load_system_prompt(bot_config):
#     config_dir = Path(__file__).parent.parent.resolve() / "config"
#
#     # load yaml config
#     with open(config_dir / "system_prompt.md", 'r') as spf:
#         system_prompt = spf.read()
#
#     # system_prompt/
#
#     try:
#         # fill the template
#         _data = {'bot_language': bot_config['bot_language']}
#         system_prompt_r = system_prompt.format(**_data)
#         system_prompt = system_prompt_r
#     except Exception as e:
#         logging.exception(e)
#
#     return system_prompt
#
#
# class AiProvider:
#     def create_completion(self, messages) -> tuple[str, int]:
#         raise NotImplementedError
#
#     def get_human_prompt(self):
#         raise NotImplementedError
#
#     def get_ai_prompt(self):
#         raise NotImplementedError
#
#
# class OpenAiProvider(AiProvider):
#     def __init__(self, config: dict, system_prompt: str):
#         self.system_prompt = system_prompt
#         self.config = config
#         http_client = httpx.AsyncClient(proxies=config['proxy']) if 'proxy' in config else None
#         self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)
#         self.model = config['model']
#
#     async def create_completion(self, messages) -> tuple[str, int]:
#         response = await self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=0.4
#         )
#         return str(response.choices[0].message), response.usage.total_tokens
#
#     def get_human_prompt(self):
#         return '\n\nUser:'
#
#     def get_ai_prompt(self):
#         return '\n\nAssistant:'
#
#
# class Claude(AiProvider):
#     def __init__(self, config: dict, system_prompt: str):
#         # self.model = config['anthropic_model']
#         self.model = "claude-3-5-sonnet-20240620"
#         self.temperature = 0.7
#         self.cutoff = 50
#         self.client = AsyncAnthropic(api_key=config['anthropic_api_key'])
#         self.prompt = ""
#         self.system_prompt = system_prompt
#
#     def get_human_prompt(self):
#         return HUMAN_PROMPT
#
#     def get_ai_prompt(self):
#         return AI_PROMPT
#
#     def remove_system_prompt(self, messages: []):
#         ret = []
#         _sys_p = None
#         for m in messages:
#             if m['role'] == 'system':
#                 _sys_p = m['content']
#             else:
#                 ret.append(m)
#         return ret, _sys_p
#
#     def coalesce_messages(self, messages: []):
#         if len(messages) == 0:
#             return messages
#
#         ret = [messages[0]]
#
#         for i in range(1, len(messages)):
#             if messages[i]['role'] == messages[i - 1]['role']:
#                 ret[-1]['content'] += '\n'
#                 ret[-1]['content'] += messages[i]['content']
#             else:
#                 ret.append(messages[i])
#
#         return ret
#
#     async def create_completion(self, messages: []) -> tuple[str, int]:
#         _messages, _sys_p = self.remove_system_prompt(messages)
#         _messages = self.coalesce_messages(_messages)
#         if _sys_p is None:
#             _sys_p = self.system_prompt
#
#         response = await self.client.messages.create(
#             model="claude-3-5-sonnet-20240620",  # TODO: get from config
#             max_tokens=1024,
#             stream=False,
#             messages=_messages,
#             system=_sys_p
#         )
#
#         answer = ''
#         if len(response.content) > 1:
#             for index, choice in enumerate(response.content):
#                 content = str(choice.text)
#
#                 answer += f'{index + 1}\u20e3\n'
#                 answer += content
#                 answer += '\n\n'
#         else:
#             answer = str(response.content[0].text)
#
#         return answer, response.usage.input_tokens + response.usage.output_tokens
#
#
# if __name__ == '__main__':
#     # test Claude
#     test_uid = 437507654
#     from dotenv import load_dotenv
#     import os
#     import asyncio
#
#     load_dotenv()
#
#     persistence = JSONFileConversationPersistence(storage_dir="logs")
#     c = persistence.load_conversation(test_uid)
#     # c, sys_p = remove_system_prompt(c)
#     # c = c[1:]
#     c = c[:-1]
#     _config = {
#         'anthropic_api_key': os.environ['ANTHROPIC_API_KEY']
#     }
#     provider = Claude(_config, load_system_prompt(_config))
#
#
#     async def msg():
#         r = await provider.create_completion(c)
#         print(r)
#
#
#     asyncio.run(msg())


from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict

import httpx
import openai
from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic

from persistence import JSONFileConversationPersistence


def load_system_prompt(bot_config: Dict[str, str]) -> str:
    """
    Load and format the system prompt from a configuration file.

    :param bot_config: Dictionary containing bot configuration
    :return: Formatted system prompt
    """
    config_dir = Path(__file__).parent.parent.resolve() / "config"

    try:
        with open(config_dir / "system_prompt.md", 'r') as prompt_file:
            system_prompt = prompt_file.read()

        return system_prompt.format(bot_language=bot_config['bot_language'])
    except Exception as e:
        logging.exception(f"Error loading system prompt: {e}")
        return ""


class AIProvider:
    """Base class for AI providers."""

    async def create_completion(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        """
        Create a completion based on the given messages.

        :param messages: List of message dictionaries
        :return: Tuple of (response text, token count)
        """
        raise NotImplementedError

    def get_human_role(self) -> str:
        return 'user'

    def get_ai_role(self) -> str:
        return 'assistant'


class OpenAIProvider(AIProvider):
    """OpenAI-specific implementation of AIProvider."""

    def __init__(self, config: Dict[str, str], system_prompt: str):
        self.system_prompt = system_prompt
        self.config = config
        http_client = httpx.AsyncClient(proxies=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)
        self.model = config['model']

    async def create_completion(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4
        )
        return str(response.choices[0].message.content), response.usage.total_tokens


class ClaudeProvider(AIProvider):
    """Claude AI-specific implementation of AIProvider."""

    def __init__(self, config: Dict[str, str], system_prompt: str):
        self.model = config.get('anthropic_model', "claude-3-5-sonnet-20240620")
        self.temperature = 0.7
        self.max_tokens = 1024
        self.client = AsyncAnthropic(api_key=config['anthropic_api_key'])
        self.system_prompt = system_prompt

    @staticmethod
    def extract_system_prompt(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        """
        Extract the system prompt from the messages and return the remaining messages.

        :param messages: List of message dictionaries
        :return: Tuple of (remaining messages, system prompt)
        """
        system_prompt = None
        remaining_messages = []

        for message in messages:
            if message['role'] == 'system':
                system_prompt = message['content']
            else:
                remaining_messages.append(message)

        return remaining_messages, system_prompt

    @staticmethod
    def merge_consecutive_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Merge consecutive messages from the same role.

        :param messages: List of message dictionaries
        :return: List of merged message dictionaries
        """
        if not messages:
            return messages

        merged = [messages[0]]

        for message in messages[1:]:
            if message['role'] == merged[-1]['role']:
                merged[-1]['content'] += f"\n{message['content']}"
            else:
                merged.append(message)

        return merged

    async def create_completion(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        messages, extracted_system_prompt = self.extract_system_prompt(messages)
        messages = self.merge_consecutive_messages(messages)

        system_prompt = extracted_system_prompt or self.system_prompt

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            system=system_prompt
        )

        if len(response.content) > 1:
            answer = "\n\n".join(f"{i + 1}\u20e3\n{str(choice.text)}" for i, choice in enumerate(response.content))
        else:
            answer = str(response.content[0].text)

        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        return answer, total_tokens


async def test_claude_provider():
    """Test function for ClaudeProvider."""
    from dotenv import load_dotenv
    import os

    load_dotenv()

    test_user_id = 437507654
    persistence = JSONFileConversationPersistence(storage_dir="logs")
    conversation = persistence.load_conversation(test_user_id)

    config = {
        'anthropic_api_key': os.environ['ANTHROPIC_API_KEY'],
        'bot_language': 'en'
    }

    system_prompt = load_system_prompt(config)
    provider = ClaudeProvider(config, system_prompt)

    response, token_count = await provider.create_completion(conversation[:-1])
    print(f"Response: {response}")
    print(f"Token count: {token_count}")


if __name__ == '__main__':
    import asyncio

    asyncio.run(test_claude_provider())
