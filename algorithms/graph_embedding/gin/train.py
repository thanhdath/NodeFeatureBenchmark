import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dataloader import GraphDataLoader, collate
from .parser import Parser
from .gin import GIN
import random
from utils import f1
import time
import os


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)

    for graphs, labels in trainloader:
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        outputs = net(graphs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(args, net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    # total_iters = len(dataloader)
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            labels = labels.to(args.device)

            total += len(labels)

            outputs = net(graphs)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()
            loss = criterion(outputs, labels)
            # crossentropy(reduce=True) for default
            total_loss += loss.item() * len(labels)

            labels = labels.cpu()

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc

def eval_net_f1(args, net, dataloader):
    net.eval()
    outputss = []
    labelss = []
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            labels = labels.to(args.device)
            outputs = net(graphs)
            outputss.append(outputs)
            labelss.append(labels)
    if len(outputss) == 0:
        return
    outputss = torch.cat(outputss, dim=0)
    labelss = torch.cat(labelss, dim=0)
    micro, macro = f1(outputss, labelss)
    print('Test micro-macro: {:.3f}\t{:.3f}'.format(micro, macro))

def gin_api(args):
    if args.cuda:
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # dataset = GINDataset(args.data, not args.learn_eps)
    # dataset = TUDataset(args.data, args)
    dataset = args.dataset
    trainloader, validloader, testloader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        collate_fn=collate).train_valid_loader()

    input_dim, label_dim, max_num_nodes = dataset.statistics()
    model = GIN(
        args.num_layers, args.num_mlp_layers,
        input_dim, args.hidden_dim, label_dim,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type,
        args.device).to(args.device)

    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    best_val_acc = 0
#     best_model_name = 'gin-best-model-{}.pkl'.format(time.time())
    best_model_name = args.model_name
    if args.transfer_from is not None:
        model.load_state_dict(torch.load(args.transfer_from))
        print("Transfer from", args.transfer_from)
    
    _, valid_acc = eval_net(args, model, validloader, criterion)
    print("== No finetuning - val acc: {:.3f}".format(valid_acc))
    
    for epoch in range(args.epochs):
        model.train()
        scheduler.step()
    
        train(args, model, trainloader, optimizer, criterion, epoch)

        train_loss, train_acc = eval_net(args, model, trainloader, criterion)
        
        _, valid_acc = eval_net(args, model, validloader, criterion)
        if epoch % 20 == 0:
            print("Epoch {} - train loss {:.3f} - train acc {:.3f} - val acc {:.3f}".format(epoch, train_loss, train_acc, valid_acc))
        if best_val_acc < valid_acc:
            best_val_acc = valid_acc
            torch.save(model.state_dict(), best_model_name)
            print("== Epoch {} - Best val acc: {:.3f}".format(epoch, valid_acc))
    model.load_state_dict(torch.load(best_model_name))
    eval_net_f1(args, model, testloader)
    # os.remove(best_model_name)



if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)
    # set up seeds, args.seed supported
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gin_api(args)
