if __name__ == '__main__':
    import os
    import datetime
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from model import DigitRecognition
    from dataset import training_data, test_data

    # Hyperparameters
    DROPOUT_RATE = 0.50
    LEARNING_RATE = 0.001
    EPOCHS = 100

    NUM_WORKERS = 16
    BATCH_SIZE = 8192

    RESUME_FROM_MODEL = None
    # RESUME_FROM_MODEL = "training_checkpoints/2023-05-31_16-30-48/epoch-20.pth";
    CHECKPOINT_DIR = "training_checkpoints"

    # Create Checkpoint directory
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    CHECKPOINT_DIR = os.path.join(
        CHECKPOINT_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(CHECKPOINT_DIR)

    # Select Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    device = torch.device(device)
    print(f"Using {device} device")

    # Load Dataset
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                                 shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)

    # Create Model
    model = DigitRecognition(DROPOUT_RATE).to(device)
    if (RESUME_FROM_MODEL):
        print("Resuming From Previously Trained Model")
        model.load_state_dict(torch.load(
            RESUME_FROM_MODEL, map_location=torch.device(device)))
    print(model)

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if batch % int(size / 5 / BATCH_SIZE) == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device, non_blocking=True), y.to(
                    device, non_blocking=True)

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Train Model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

        if (t + 1) % 10 == 0 or (t + 1) == EPOCHS:
            torch.save(model.state_dict(), os.path.join(
                CHECKPOINT_DIR, "epoch-" + str(t + 1) + ".pth"))
    print("Done Training!")
