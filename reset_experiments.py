"""
Run this before each new experiment strategy run, to guarantee a clean,
fair comparison -- wipes long-term memory for the experiment user.
Short-term reset is still manual: just change THREAD_ID in main.py.
"""

from app.memory.long_term import LongTermMemory

USER_ID = "sahil"   # keep in sync with main.py

if __name__ == "__main__":
    memory = LongTermMemory()
    memory.delete_all(USER_ID)
    print(f"Long-term memory cleared for user_id={USER_ID}")