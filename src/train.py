import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

def train(model, dataset, epochs=50, batch_size=32, lr=1e-3, device="cpu"):
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Move model to device
    model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            # Move data to device
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.size(-1)), 
                tgt[:, 1:].reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        checkpoint_path = f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved as {checkpoint_path}")