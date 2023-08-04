import torch
import utils

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device = None):
    
    # If device is not provided, use the default device (CPU)
    if device is None:
        device = torch.device("cpu")

    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager, this is the same as torc.no_grad
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            output = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(output, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_acc  += utils.accuracy(output, y)

            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc