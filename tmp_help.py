import malaya
import sys

with open("malaya_help.txt", "w", encoding="utf-8") as f:
    orig_stdout = sys.stdout
    sys.stdout = f
    help(malaya.summarization.extractive)
    sys.stdout = orig_stdout
