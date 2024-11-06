import time

import torch
import torch.nn.functional as F


def train_step(model, images, targets, optimizer, device):
    model.train()
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    y_true = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

    # Forward pass
    outputs = model(images)

    # Compute the loss
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for output, target in zip(outputs, y_true):
        pred_logits = output['logits']
        true_labels = target['labels']
        loss = loss_fn(pred_logits, true_labels)
        total_loss += loss

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), outputs


def train(model, train_loader, optimizer, lr_scheduler, num_epochs, device, writer, output_dir):
    prev_lr = 0
    prev_loss = 0
    total_steps = len(train_loader) * num_epochs

    for epoch in range(num_epochs):
        model.train()
        epoch_timer = time.time()
        print(f"Beginning epoch {epoch + 1}/{num_epochs}...")
        for step, (images, targets) in enumerate(train_loader):
            total_loss, outputs = train_step(model, images, targets, optimizer, device)

            current_global_step = epoch * len(train_loader) + step
            writer.add_scalar("Loss/total_loss", total_loss, current_global_step)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], current_global_step)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Step {current_global_step}/{total_steps}:")
            print(f"Total loss: {total_loss}\nLearning rate: {optimizer.param_groups[0]['lr']}")

            next_loss = total_loss
            delta_loss = prev_loss - next_loss
            print(f"Training loss delta: {delta_loss}")
            prev_loss = next_loss

            next_lr = optimizer.param_groups[0]["lr"]
            delta_lr = prev_lr - next_lr
            print(f"Learning rate delta: {delta_lr}")
            prev_lr = next_lr

        print(f"Epoch [{epoch + 1}/{num_epochs}] complete! Took {time.time() - epoch_timer:.3f} seconds")
        lr_scheduler.step()

    writer.close()
    print("Training complete!")
