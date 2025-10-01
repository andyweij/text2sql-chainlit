import os
import json
import bcrypt

ACCOUNTS_DIR = ".chainlit"
ACCOUNTS_FILE = os.path.join(ACCOUNTS_DIR, ".accounts.jsonl")

def load_accounts():
    print("accounts")
    """讀取 .accounts.jsonl 檔案，回傳帳號列表，每一筆資料是一個 dict"""
    accounts = []
    if os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    account = json.loads(line)
                    accounts.append(account)
    return accounts

def authenticate_account(identifier: str, password: str):
    accounts = load_accounts()
    account = next((acc for acc in accounts if acc.get("identifier") == identifier), None)

    if account is None:
        return None
    else:
        stored_hashed = account.get("password", "")
        if bcrypt.checkpw(password.encode("utf-8"), stored_hashed.encode("utf-8")):
            return {
                "identifier": identifier,
                "role": account.get("role", None),
                "provider": account.get("provider", None)
            }

    return None
