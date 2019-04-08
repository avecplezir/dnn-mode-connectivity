import torch
import utils
import models
import argparse

parser = argparse.ArgumentParser(description='find expectation')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--ind_beg', type=int, default=1)
parser.add_argument('--ind_end', type=int, default=10)

args = parser.parse_args()

architecture = getattr(models, args.model)
model = architecture.base(num_classes=10, **architecture.kwargs)

model = architecture.base(num_classes=10, **architecture.kwargs)
m = architecture.base(num_classes=10, **architecture.kwargs)


for i, ind in enumerate(range(args.ind_beg, args.ind_end)):

    ckpt = args.ckpt+'/curve' + str(2 + ind) + '/checkpoint-12.pt'
    checkpoint = torch.load(ckpt)
    m.load_state_dict(checkpoint['model_state'])

    for parameter, p in zip(model.parameters(), m.parameters()):
        if i == 0:
            parameter.data.copy_((p))
        else:
            parameter.data.copy_((parameter + p))

N = args.ind_end - args.ind_beg
for parameter in model.parameters():
    parameter.data.copy_(parameter / N)

checkpoints = torch.load('curves_mnist/LinearOneLayer/curve3/checkpoint-12.pt')

print("Saving checkpoint for node changing")

utils.save_checkpoint(
    args.dir,
    100,
    name='E',
    model_state=model.state_dict(),
    optimizer_state=checkpoints['optimizer_state']
)
