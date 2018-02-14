import argparse
import json
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm
from tqdm import trange
from warpctc_pytorch import CTCLoss
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Size of input')
parser.add_argument('--seconds', type=int, default=5,
                    help='The size of the fake input in seconds using default stride of 0.01, '
                         '15s is usually the maximum duration')
parser.add_argument('--num-samples', type=int, default=1024, help='Number of samples to replicate')
parser.add_argument('--dry-runs', type=int, default=2, help='Dry runs before measuring performance')
parser.add_argument('--runs', type=int, default=5, help='How many benchmark runs to measure performance')
parser.add_argument('--labels-path', default='labels.json', help='Path to the labels to infer over in the model')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--half', action="store_true", help='Use half precision')
args = parser.parse_args()


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(161, args.hidden_size, num_layers=args.hidden_layers)
        self.fc = SequenceWise(torch.nn.Linear(args.hidden_size, len(labels)))

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(x)


class tofp16(torch.nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def BN_convert_float(module):
    """
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return torch.nn.Sequential(tofp16(), BN_convert_float(network.half()))


def set_grad(param_copy, param):
    for p_optim, p_model in zip(param_copy, param):
        p_optim.grad = p_model.grad.float()


if __name__ == '__main__':
    inputs = torch.randn(args.num_samples, args.seconds * 100, 161).cuda()
    inputs = torch.chunk(inputs, int(len(inputs) / args.batch_size))

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    model = SimpleModel()
    model = model.cuda()

    if args.half:
        param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
        for param in param_copy:
            param.requires_grad = True
        optimizer = torch.optim.SGD(param_copy, lr=3e-4, momentum=0.9, nesterov=True)
        model = network_to_half(model)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, nesterov=True)
    model = torch.nn.DataParallel(model)
    criterion = CTCLoss()

    seconds = int(args.seconds)
    batch_size = int(args.batch_size)


    def iteration(input_data):
        model.train()
        input_data = input_data.transpose(0, 1)  # TxNxH
        target = torch.IntTensor(int(batch_size * ((seconds * 100) / 2))).fill_(1)  # targets, align half of the audio
        target_size = torch.IntTensor(batch_size).fill_(int((seconds * 100) / 2))
        input_percentages = torch.IntTensor(batch_size).fill_(1)

        inputs = Variable(input_data, requires_grad=False)
        target_sizes = Variable(target_size, requires_grad=False)
        targets = Variable(target, requires_grad=False)
        out = model(inputs)
        seq_length = out.size(0)
        sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)
        out = out.cuda().float()
        loss = criterion(out, targets, sizes, target_sizes)
        loss = loss / inputs.size(0)  # average the loss by minibatch
        # compute gradient
        model.zero_grad()
        loss.backward()
        if args.half:
            set_grad(param_copy, list(model.parameters()))
            optimizer.step()
            params = list(model.parameters())
            for x in range(len(params)):
                params[x].data.copy_(param_copy[x].data)
        else:
            optimizer.step()
        torch.cuda.synchronize()
        del loss
        del out


    print("Running dry runs...")
    for n in trange(args.dry_runs):
        for data in tqdm(inputs, total=len(inputs)):
            iteration(data)

    print("\n Running measured runs...")
    running_time = 0
    for n in trange(args.runs):
        start_time = time.time()
        for data in tqdm(inputs, total=len(inputs)):
            iteration(data)
        end_time = time.time()
        running_time += (end_time - start_time)

    run_time = running_time / float(args.runs)

    print("\n Average run time: %.2fs" % run_time)
