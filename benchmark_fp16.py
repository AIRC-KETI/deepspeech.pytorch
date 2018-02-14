import argparse
import json
import time
import torch
from torch.autograd import Variable
from tqdm import tqdm
from warpctc_pytorch import CTCLoss
from tqdm import trange

from data.utils import network_to_half, set_grad
from model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, help='Size of input')
parser.add_argument('--seconds', type=int, default=15,
                    help='The size of the fake input in seconds using default stride of 0.01, '
                         '15s is usually the maximum duration')
parser.add_argument('--num-samples', type=int, default=1024, help='Number of samples to replicate')
parser.add_argument('--dry-runs', type=int, default=2, help='Dry runs before measuring performance')
parser.add_argument('--runs', type=int, default=5, help='How many benchmark runs to measure performance')
parser.add_argument('--labels-path', default='labels.json', help='Path to the labels to infer over in the model')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
args = parser.parse_args()

input_data = torch.randn(args.num_samples, 1, 161, args.seconds * 100).cuda()
input_data = torch.chunk(input_data, int(len(input_data) / args.batch_size))
rnn_type = args.rnn_type.lower()
assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

with open(args.labels_path) as label_file:
    labels = str(''.join(json.load(label_file)))

audio_conf = dict(sample_rate=args.sample_rate,
                  window_size=args.window_size)

model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                   nb_layers=args.hidden_layers,
                   audio_conf=audio_conf,
                   labels=labels,
                   rnn_type=supported_rnns[rnn_type])
print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
for param in param_copy:
    param.requires_grad = True
optimizer = torch.optim.SGD(param_copy, lr=3e-4, momentum=0.9, nesterov=True)

model = network_to_half(model)
model = torch.nn.DataParallel(model).cuda()
criterion = CTCLoss()

seconds = int(args.seconds)
batch_size = int(args.batch_size)


def iteration(input_data):
    model.train()
    target = torch.IntTensor(int(batch_size * ((seconds * 100) / 2))).fill_(1)  # targets, align half of the audio
    target_size = torch.IntTensor(batch_size).fill_(int((seconds * 100) / 2))
    input_percentages = torch.IntTensor(batch_size).fill_(1)

    inputs = Variable(input_data, requires_grad=False)
    target_sizes = Variable(target_size, requires_grad=False)
    targets = Variable(target, requires_grad=False)
    out = model(inputs)
    out = out.transpose(0, 1)  # TxNxH

    seq_length = out.size(0)
    sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)
    out = out.cuda().float()
    loss = criterion(out, targets, sizes, target_sizes)
    loss = loss / inputs.size(0)  # average the loss by minibatch
    # compute gradient
    model.zero_grad()
    loss.backward()
    set_grad(param_copy, list(model.parameters()))
    optimizer.step()
    params = list(model.parameters())
    for x in range(len(params)):
        params[x].data.copy_(param_copy[x].data)
    torch.cuda.synchronize()
    del loss
    del out


def run_benchmark():
    print("Running dry runs...")
    for n in trange(args.dry_runs):
        for data in tqdm(input_data, total=len(input_data)):
            iteration(data)

    print("\n Running measured runs...")
    running_time = 0
    for n in trange(args.runs):
        start_time = time.time()
        for data in tqdm(input_data, total=len(input_data)):
            iteration(data)
        end_time = time.time()
        running_time += (end_time - start_time)

    return running_time / float(args.runs)


run_time = run_benchmark()

print("\n Average run time: %.2fs" % run_time)

