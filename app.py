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


def generate_text(input, model, num_sequences, max_length):
    tokens = TOKENIZER.encode(input)
    tokens = torch.tensor(tokens, dtype=torch.long) 
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1) 
    x = tokens.to(device)

    sentences = []
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1) 
            xcol = torch.gather(topk_indices, -1, ix) 
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = TOKENIZER.decode(tokens)
        sentences.append(decoded) 

    return sentences


def gradio_fn(prompt, num_sequences=1, max_length=30):
    """Generate text using both models."""
    untrained_texts = generate_text(prompt, UNTRAINED_MODEL, num_sequences, max_length)
    untrained_output = "\n\n".join(f"> {s}" for s in untrained_texts)

    trained_texts = generate_text(prompt, TRAINED_MODEL, num_sequences, max_length)
    trained_output = "\n\n".join(f"> {s}" for s in trained_texts)

    return untrained_output, trained_output

# Gradio interface
def main():
    interface = gr.Interface(
        fn=gradio_fn,
        inputs=[
            gr.Textbox(label="Enter your prompt here:"),
            gr.Slider(minimum=1, maximum=10, step=1, label="Number of Generations"),
            gr.Slider(minimum=10, maximum=100, step=10, label="Max Length"),
        ],
        outputs=[
            gr.Textbox(label="Generated Text (Untrained Model)"),
            gr.Textbox(label="Generated Text (Trained Model)"),
        ],
        title="GPT-2  Text Generator",
        description="Generate text an untrained and a trained GPT-2 model."
    )

    interface.launch(share=True)

if __name__ == "__main__":
    main()


