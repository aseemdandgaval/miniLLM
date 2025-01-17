import torch
import tiktoken
import gradio as gr
from model import GPT, GPTConfig
from torch.nn import functional as F

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer
TOKENIZER = tiktoken.get_encoding('gpt2')

# Load untrained model
UNTRAINED_MODEL = GPT(GPTConfig)
UNTRAINED_MODEL.to(device)
UNTRAINED_MODEL.eval()

# Load fine-tuned model 
TRAINED_MODEL = GPT(GPTConfig)
checkpoint = torch.load("log/model_19072.pt", weights_only=False)
TRAINED_MODEL.load_state_dict(checkpoint["model"])
TRAINED_MODEL.to(device)
TRAINED_MODEL.eval()


def generate_text(input, max_length=30, top_k=50):
    tokens = TOKENIZER.encode(input)
    x_untrained = torch.tensor([tokens], dtype=torch.long).to(device)
    x_trained   = torch.tensor([tokens], dtype=torch.long).to(device)

    # Iterate until one of the sequences reaches max_length
    while (x_untrained.size(1) < max_length) or (x_trained.size(1) < max_length):

        # --- Untrained Model Forward Pass ---
        if x_untrained.size(1) < max_length:
            with torch.no_grad():
                logits_u, _ = UNTRAINED_MODEL(x_untrained)
                logits_u = logits_u[:, -1, :]
                probs_u = F.softmax(logits_u, dim=-1)
                topk_probs_u, topk_indices_u = torch.topk(probs_u, top_k, dim=-1)
                ix_u = torch.multinomial(topk_probs_u, 1)
                next_token_u = torch.gather(topk_indices_u, -1, ix_u)
            x_untrained = torch.cat((x_untrained, next_token_u), dim=1)

        # --- Trained Model Forward Pass ---
        if x_trained.size(1) < max_length:
            with torch.no_grad():
                logits_t, _ = TRAINED_MODEL(x_trained)
                logits_t = logits_t[:, -1, :]
                probs_t = F.softmax(logits_t, dim=-1)
                topk_probs_t, topk_indices_t = torch.topk(probs_t, top_k, dim=-1)
                ix_t = torch.multinomial(topk_probs_t, 1)
                next_token_t = torch.gather(topk_indices_t, -1, ix_t)
            x_trained = torch.cat((x_trained, next_token_t), dim=1)

        # --- Decode the partial text for each model ---
        untrained_text = TOKENIZER.decode(x_untrained[0].tolist())
        trained_text   = TOKENIZER.decode(x_trained[0].tolist())

        yield (untrained_text, trained_text)


def streaming_fn(prompt, max_length=30, top_k=50):
    for untrained_text, trained_text in generate_text(prompt, max_length, top_k):
        output = (
            f"------------ (Untrained Model) ------------\n\n {untrained_text}\n\n\n"
            f"------------ (Trained Model)------------\n\n {trained_text}"
            )
        yield output
    

def main():
    interface = gr.Interface(
        fn=streaming_fn,
        inputs=[
            gr.Textbox(label="Enter your prompt here:"),
            gr.Slider(minimum=10, maximum=150, step=10, label="Max Length"),
            gr.Slider(minimum=1, maximum=50, step=10, label="Top-K Samples")
        ],
        outputs=gr.Textbox(label="Model Outputs"),
        title="GPT-2 Streaming Text Generator",
        description= (
            "Generate text using an untrained and a trained GPT-2 model."
            "Use prompts that are short, simple and easy to generate coherent looking text."
            "For eg: \n"
            "- \"Hello, my name is\" \n"
            "- \"This is a summary of\" \n"
            "- \"In this article\" \n"
        )
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()