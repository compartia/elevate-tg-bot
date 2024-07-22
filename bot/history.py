from typing import List, Dict
import json


class ChatHistory:
    def __init__(self, chat_id: int, persistence):
        self.chat_id = chat_id
        self.messages: List[Dict[str, str]] = []
        self.persistence = persistence
        self.load()

    def add(self, role: str, content: str):
        """
        Add a new message to the chat history.

        :param role: The role of the message sender (e.g., 'user', 'assistant', 'system')
        :param content: The content of the message
        """
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.save()

    def trim(self, max_len:int):
        self.messages = self.messages[-max_len:]
    def save(self):
        """
        Save the current chat history to the persistence layer.
        """
        self.persistence.save_conversation(self.chat_id, self.messages)

    def load(self):
        """
        Load the chat history from the persistence layer.
        """
        self.messages = self.persistence.load_conversation(self.chat_id)

    def clear(self):
        """
        Clear the chat history.
        """
        self.messages = []
        self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in the chat history.

        :return: List of message dictionaries
        """
        return self.messages

    def __len__(self):
        """
        Get the number of messages in the chat history.

        :return: Number of messages
        """
        return len(self.messages)

    def __getitem__(self, index):
        """
        Get a message by index.

        :param index: Index of the message
        :return: Message dictionary
        """
        return self.messages[index]

    def __iter__(self):
        """
        Iterate over the messages in the chat history.

        :return: Iterator of message dictionaries
        """
        return iter(self.messages)

    def __str__(self):
        """
        Get a string representation of the chat history.

        :return: String representation of the chat history
        """
        return json.dumps(self.messages, indent=2)