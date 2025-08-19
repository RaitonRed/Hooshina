import torch

def chat(model, tokenizer, device="cpu", max_len=64):
    model.eval()
    while True:
        user_input = input("👤 you: ")
        if user_input.lower() in ["exit", "quit", "خروج"]:
            break

        src = torch.tensor([tokenizer.encode(user_input, max_len)], device=device)
        tgt = torch.tensor([[tokenizer.word2id["<pad>"]]], device=device)

        for _ in range(max_len - 1):
            output = model(src, tgt)
            next_token = output.argmax(-1)[:, -1]
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.word2id["<pad>"]:
                break

        print("🤖 bot: ", tokenizer.decode(tgt.squeeze().tolist()))
