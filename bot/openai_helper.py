from __future__ import annotations

import datetime
import json
import logging
import os

import httpx
import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from ai_provider import AIProvider
from bot.history import ChatHistory
from persistence import ConversationPersistence

# Models can be found here: https://platform.openai.com/docs/models/overview
# Models gpt-3.5-turbo-0613 and  gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
GPT_3_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613")
GPT_3_16K_MODELS = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125")
GPT_4_MODELS = ("gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-turbo-preview")
GPT_4_32K_MODELS = ("gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613")
GPT_4_VISION_MODELS = ("gpt-4-vision-preview",)
GPT_4_128K_MODELS = (
    "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-turbo-2024-04-09")
GPT_4O_MODELS = ("gpt-4o",)
GPT_ALL_MODELS = GPT_3_MODELS + GPT_3_16K_MODELS + GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_VISION_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS


def default_max_tokens(model: str) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1200
    if model in GPT_3_MODELS:
        return base
    elif model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_3_16K_MODELS:
        if model == "gpt-3.5-turbo-1106":
            return 4096
        return base * 4
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_VISION_MODELS:
        return 4096
    elif model in GPT_4_128K_MODELS:
        return 4096
    elif model in GPT_4O_MODELS:
        return 4096


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    # Deprecated models
    if model in ("gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-32k-0314"):
        return False
    # Stable models will be updated to support functions on June 27, 2023
    if model in (
            "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "gpt-4-0125-preview",
            "gpt-4-turbo-preview"):
        return datetime.date.today() > datetime.date(2023, 6, 27)
    # Models gpt-3.5-turbo-0613 and  gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
    if model in ("gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"):
        return datetime.date.today() < datetime.date(2024, 6, 13)
    if model == 'gpt-4-vision-preview':
        return False
    return True


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    # TODO: translate via bot.

    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations['en']:
            return translations['en'][key]
        else:
            logging.warning(f"No english definition found for key '{key}' in translations.json")
            # return key as text
            return key


class AIHelper:
    """
    ChatGPT helper class.
    """

    def __init__(self, config: dict, persistence: ConversationPersistence, provider: AIProvider):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        """
        self.persistence = persistence

        http_client = httpx.AsyncClient(proxies=config['proxy']) if 'proxy' in config else None
        # self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)
        self.provider = provider
        self.config = config
        self.conv: dict[int: ChatHistory] = {}  # {chat_id: history}

        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, int]:
        """
        Gets a full response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """

        answer, total_tokens = await self.__common_get_chat_response(chat_id, query)

        role = self.provider.get_ai_role()
        self.conv[chat_id].add(role, answer)
        return answer, total_tokens

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response(self, chat_id: int, query: str):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conv or self.__max_age_reached(chat_id):
                # TODO: load
                self.init_conversation(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            # TODO: to avoid double role, add it only after response received
            self.conv[chat_id].add(self.provider.get_human_role(), query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conv[chat_id].get_messages())
            exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conv[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conv[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_conversation(chat_id)
                    self.conv[chat_id].add(self.provider.get_ai_role(), summary)
                    self.conv[chat_id].add(self.provider.get_human_role(), query)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conv[chat_id].trim(self.config['max_history_size'])

            messages = self.conv[chat_id].get_messages()
            return await self.provider.create_completion(messages)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    def init_conversation(self, chat_id, content=''):
        """
        Loads the conversation history.
        """
        if chat_id not in self.conv:
            self.conv[chat_id] = ChatHistory(chat_id, self.persistence)
        # self.conv[chat_id].clear()

    def reset_conversation(self, chat_id, content=''):
        """
        Resets the conversation history.
        """
        if chat_id not in self.conv:
            self.conv[chat_id] = ChatHistory(chat_id, self.persistence)
        self.conv[chat_id].clear()

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    async def __summarise(self, conversation: ChatHistory) -> str:
        # """
        # Summarises the conversation history.
        # :param conversation: The conversation history
        # :return: The summary
        # """
        # messages = [
        #     {"role": "assistant", "content": "Summarize this conversation in 700 characters or less"},
        #     {"role": "user", "content": str(conversation)}
        # ]
        # response = await self.client.chat.completions.create(
        #     model=self.config['model'],
        #     messages=messages,
        #     temperature=0.4
        # )
        #
        # return self.provider.create_completion(messages)
        #
        # return response.choices[0].message.content

        # TODO: implement
        logging.error('_summarize is not implemented!')
        print('_summarize is not implemented!')
        return ""

    def __max_model_tokens(self):
        base = 4096
        if self.config['model'] in GPT_3_MODELS:
            return base
        if self.config['model'] in GPT_3_16K_MODELS:
            return base * 4
        if self.config['model'] in GPT_4_MODELS:
            return base * 2
        if self.config['model'] in GPT_4_32K_MODELS:
            return base * 8
        if self.config['model'] in GPT_4_VISION_MODELS:
            return base * 31
        if self.config['model'] in GPT_4_128K_MODELS:
            return base * 31
        if self.config['model'] in GPT_4O_MODELS:
            return base * 31
        raise NotImplementedError(
            f"Max tokens for model {self.config['model']} is not implemented yet."
        )

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def __count_tokens(self, messages) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        model = self.config['model']
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("gpt-3.5-turbo")

        if model in GPT_3_MODELS + GPT_3_16K_MODELS:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model in GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_VISION_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content':
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        for message1 in value:
                            if not (message1['type'] == 'image_url'):
                                num_tokens += len(encoding.encode(message1['text']))
                else:
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # no longer needed
