import torch
import torch.nn.functional as F
import torch.optim as optim
from examples.tutorial_01_tensorboard_mnist.mnist import model
from examples.tutorial_01_tensorboard_mnist.mnist.dataloader import (
    test_loader,
    train_loader
)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

epochs = 5
lr = 0.01
momentum = 0.5
seed = 1
save_model = True


use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        avg_loss += F.nll_loss(output, target, reduction="sum").item()

        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )

    accuracy = 100.0 * correct / len(train_loader.dataset)
    avg_loss /= len(train_loader.dataset)

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

    return avg_loss


def model_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # noinspection PyPackageRequirements
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    accuracy = 100.0 * correct / len(test_loader.dataset)

    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)

    return test_loss, accuracy


train_losses = []
test_losses = []
accuracy_list = []
for epoch in range(1, epochs + 1):
    trn_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss, accuracy = model_test(model, device, test_loader)
    train_losses.append(trn_loss)
    test_losses.append(test_loss)
    accuracy_list.append(accuracy)


writer.close()
