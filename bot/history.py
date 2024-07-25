from __future__ import annotations

from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage

from persistence import ConversationPersistence


class PersistentChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, chat_id: int, persistence: ConversationPersistence):
        super().__init__()
        self.chat_id = chat_id
        self.messages: List[BaseMessage] = []
        self.persistence: ConversationPersistence = persistence

        self.load()

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store.

        Args:
            message: The message to add.
        """
        self.messages.append(message)
        self.save()

    def clear(self) -> None:
        super().clear()
        self.save()

    def save(self) -> None:
        messages = [{"role": m.type, "content": m.content} for m in self.messages]
        self.persistence.save_conversation(self.chat_id, messages)

    def load(self) -> None:
        messages = self.persistence.load_conversation(self.chat_id)
        for message in messages:
            if message["role"] in ["human", "user"]:
                self.add_user_message(message["content"])
            elif message["role"] in ["ai", "assitant"]:
                self.add_ai_message(message["content"])
            elif message["role"] == "system":
                self.add_message(SystemMessage(content=message["content"]))
