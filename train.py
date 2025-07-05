import os
import subprocess

# Define all 20 combinations of model and attention
models = ['VanillaRNN', 'VanillaLSTM', 'BidirectionalRNN', 'BidirectionalLSTM']
attentions = ['None', 'Bahdanau', 'LuongDot', 'LuongGeneral', 'LuongConcat']


os.makedirs("saved_models", exist_ok=True)
os.makedirs("attention_outputs", exist_ok=True)


GLOVE_PATH = "glove.6B.100d.txt"


EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001


for model in models:
    for attention in attentions:
        print(f"\n===== Training {model} with {attention} Attention =====")

        # Define command
        cmd = [
            "python", "main.py",
            "--model", model,
            "--attention", attention,
            "--glove_path", GLOVE_PATH,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR)
        ]

        # Output file for logs
        log_file = f"logs/{model}_{attention}.log"
        os.makedirs("logs", exist_ok=True)

        with open(log_file, "w") as log:
            subprocess.run(cmd, stdout=log, stderr=log)

        print(f"Finished training {model} with {attention}.")
        print(f"Log saved to {log_file}\n")

