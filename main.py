import torch
from torch import nn, optim
from data.data import load_data
from model.transformer import Transformer
from utils.train_eval import train_model, evaluate_model

def main():
    """
    Main function to train and evaluate the Transformer model.
    """
    # Hyperparameters
    num_layers = 2
    d_model = 512
    num_heads = 8
    dff = 2048
    input_vocab_size = 1000
    target_vocab_size = 1000
    max_positional_encoding = 1000
    dropout = 0.1
    batch_size = 8
    num_epochs = 10

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_loader = load_data(batch_size)

    # Model, loss function, and optimizer
    model = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_positional_encoding, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, data_loader, criterion, optimizer, device)
        eval_loss = evaluate_model(model, data_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Eval Loss: {eval_loss}")

if __name__ == "__main__":
    main()
