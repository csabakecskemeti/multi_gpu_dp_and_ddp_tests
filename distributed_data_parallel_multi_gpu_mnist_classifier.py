import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os
from time import time

def ddp_setup(rank, world_size):
    """
    :param rank: Unique identifier for each process
    :param world_size: Total number of processes (GPUs?)
    :return:
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


# create network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data_loader: DataLoader,
            test_data_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        source = source.reshape(source.shape[0], -1)
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        # loss =  nn.CrossEntropyLoss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data_loader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data_loader)}")
        i = 0
        batch_size = len(self.train_data_loader)
        for source, targets in self.train_data_loader:
            i += 1
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            # print(i%batch_size)
            # if i%batch_size == 0:
                # print(f"loss: {loss}")
        # check_accuracy(self.model, self.test_data_loader, self.gpu_id)


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_checkpoint(epoch)


def load_train_objs(input_size, num_classes):
    train_set = datasets.MNIST(root='datasets/', train = True, transform = transforms.ToTensor(), download = True)
    testset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
    model = NN(input_size=input_size, num_classes=num_classes)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_set, testset, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


# Check accuracy
def check_accuracy(model, loader, gpu_id):
    if loader.dataset.train:
        print("Chk accuracy on training data")
    else:
        print("Chk accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            #             print(x.dtype)
            x = x.to(gpu_id)
            y = y.to(gpu_id)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    acc = float(num_correct) / float(num_samples) * 100
    model.train()
    return acc


def main(rank: int,
         world_size: int,
         total_epochs: int,
         save_every: int,
         batch_size: int,
         input_size: int,
         num_classes: int,
         profiler: bool):
    ddp_setup(rank, world_size)
    dataset, testset, model, optimizer = load_train_objs(input_size, num_classes)
    train_data_loader = prepare_dataloader(dataset, batch_size)
    test_data_loader = prepare_dataloader(testset, batch_size)
    trainer = Trainer(model, train_data_loader, test_data_loader, optimizer, rank, save_every)
    print(profiler)
    if profiler is True:
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            trainer.train(total_epochs)
        print(prof)
    else:
        start = time()
        trainer.train(total_epochs)
        end = time()
        print("Finish with:{} second".format(end - start))
    check_accuracy(trainer.model, test_data_loader, rank)
    PATH = './mnist_ddp_mult_gpu.pth'
    torch.save(trainer.model.state_dict(), PATH)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--profiler', dest='profiler', action='store_true', help='Use autograd profiler')
    parser.set_defaults(profiler=False)
    args = parser.parse_args()

    input_size = 28 * 28
    num_classes = 10
    # batch_size = 64
    # num_epochs = 1
    lrn_rate = 0.001

    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    # world_size = 1
    print("GPUs available: " + str(world_size))
    mp.spawn(main,
             args=(world_size, args.total_epochs, args.save_every, args.batch_size, input_size, num_classes, args.profiler),
             nprocs=world_size)

