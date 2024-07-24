import json
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


class ConversationPersistence:
    def load_conversation(self, chat_id: int) -> list:
        raise NotImplementedError

    def save_conversation(self, chat_id: int, conversation: list):
        raise NotImplementedError

class IdempotentPersistence(ConversationPersistence):
    def load_conversation(self, chat_id: int) -> list:
        pass

    def save_conversation(self, chat_id: int, conversation: list):
        pass


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


class FirebaseConversationPersistence(ConversationPersistence):
    def __init__(self):
        super().__init__()
        # Load Firebase credentials from environment variables
        cred_dict = {
            "type": "service_account",
            "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
            "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
            "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL"),
            "universe_domain": "googleapis.com"
        }

        self.collection = os.environ.get("FIREBASE_DATABASE_COLLECTION", 'test_conversations')
        # Initialize Firebase app
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.environ.get("FIREBASE_DATABASE_URL")
        })
        self.root = db.reference()

    def load_conversation(self, chat_id: int) -> list:
        conversation_ref = self.root.child(self.collection).child(str(chat_id))
        conversation_data = conversation_ref.get()
        return conversation_data if conversation_data else []

    def save_conversation(self, chat_id: int, conversation: list):
        conversation_ref = self.root.child(self.collection).child(str(chat_id))
        conversation_ref.set(conversation)


if __name__ == '__main__':
    # TEST, TODO:remove
    test_uid = 437507654
    from dotenv import load_dotenv

    load_dotenv()
    persistence = FirebaseConversationPersistence()
    persistence_j = JSONFileConversationPersistence(storage_dir="logs")

    c = persistence.load_conversation(test_uid)

    print(c)
    persistence.save_conversation(-test_uid, c)
