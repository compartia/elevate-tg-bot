import json
import os


class ConversationPersistence:
    def load_conversation(self, chat_id: int) -> list:
        raise NotImplementedError

    def save_conversation(self, chat_id: int, conversation: list):
        raise NotImplementedError


class JSONFileConversationPersistence(ConversationPersistence):
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def load_conversation(self, chat_id: int) -> list:
        file_path = os.path.join(self.storage_dir, f"{chat_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_conversation(self, chat_id: int, conversation: list):
        file_path = os.path.join(self.storage_dir, f"{chat_id}.json")
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    persistence = JSONFileConversationPersistence(storage_dir="logs")
    c = persistence.load_conversation(437507654)

    print(c)
    persistence.save_conversation(0, c)
