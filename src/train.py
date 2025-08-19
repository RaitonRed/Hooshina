import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

def train(model, dataset, epochs=50, batch_size=32, lr=1e-3, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        print(f"Model saved as model_epoch_{epoch+1}.pth")