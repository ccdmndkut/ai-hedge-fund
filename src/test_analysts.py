from utils.analysts import ANALYST_ORDER

print("Available analysts:")
for display, key in ANALYST_ORDER:
    print(f"- {display} ({key})") 