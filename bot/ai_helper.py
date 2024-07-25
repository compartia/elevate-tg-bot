from __future__ import annotations

import json
import logging
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from history import PersistentChatMessageHistory
from persistence import ConversationPersistence

# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        return translations.get('en', {}).get(key, key)


class AIHelper:
    def __init__(self,  persistence: ConversationPersistence, model: BaseChatModel = None, prompt=""):

        self.persistence = persistence
        self.prompt = prompt
        self.model = model  # or ChatAnthropic(model_name="claude-3-sonnet-20240229")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.chain = self.prompt | self.model
        self.histories = {}

        def get_session_history(session_id: int) -> PersistentChatMessageHistory:
            h = self.histories.get(session_id, PersistentChatMessageHistory(session_id, self.persistence))
            self.histories[session_id] = h
            return h

        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, int]:
        try:
            response = await self.chain_with_history.ainvoke(
                {"input": query},
                config={"configurable": {"session_id": chat_id}}
            )

            ai_message = response.content

            # Estimate token usage (this is a rough estimate)
            total_tokens = len(query.split()) + len(ai_message.split())

            return ai_message, total_tokens
        except Exception as e:
            logging.error(f"Error in get_chat_response: {str(e)}")
            raise

    # def reset_conversation(self, chat_id: int):
    #     memory = self.chain_with_history.lookup_memory(chat_id)
    #     memory.clear()

    def reset_conversation(self, chat_id: int) -> None:
        self.histories[chat_id] = PersistentChatMessageHistory(chat_id, self.persistence)
        self.histories[chat_id].clear()

    # Placeholder for future summarization feature
    async def summarize(self, chat_id: int) -> str:
        # This will be implemented in the future
        return "Summarization not implemented yet."
