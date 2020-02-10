from __future__ import print_function
import argparse

import torch
from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='/data/yusun/datasets/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--grad_corr', action='store_true')
parser.add_argument('--visualize_samples', action='store_true')
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--resume', default=None)
parser.add_argument('--none', action='store_true')

args = parser.parse_args()
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)
teset, teloader = prepare_test_data(args) # teloader在这里得到，所以 prepare_test_data 是一个关键函数

# print(net.state_dict().keys())

print('Resuming from %s...' %(args.resume))
ckpt = torch.load(args.resume + '/ckpt.pth')
net.load_state_dict(ckpt['net']) # 这个地方报错了
cls_initial, cls_correct, cls_losses = test(teloader, net) # 这一步说明了是在什么任务上做测试

print('Old test error cls %.2f' %(ckpt['err_cls']*100))
print('New test error cls %.2f' %(cls_initial*100)) # 这个测试是在发生变化的图片上的测试

if args.none:
	rdict = {'cls_initial': cls_initial, 'cls_correct': cls_correct, 'cls_losses': cls_losses}
	torch.save(rdict, args.outf + '/%s_%d_none.pth' %(args.corruption, args.level))
	quit()

print('Old test error ssh %.2f' %(ckpt['err_ssh']*100)) # 为什么没有 new
head.load_state_dict(ckpt['head'])
ssh_initial, ssh_correct, ssh_losses = [], [], []

labels = [0,1,2,3]
for label in labels:
	tmp = test(teloader, ssh, sslabel=label)
	ssh_initial.append(tmp[0])
	ssh_correct.append(tmp[1])
	ssh_losses.append(tmp[2])

rdict = {'cls_initial': cls_initial, 'cls_correct': cls_correct, 'cls_losses': cls_losses,
			'ssh_initial': ssh_initial, 'ssh_correct': ssh_correct, 'ssh_losses': ssh_losses}
torch.save(rdict, args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))

if args.grad_corr:
	corr = test_grad_corr(teloader, net, ssh, ext)
	print('Average gradient inner product: %.2f' %(mean(corr)))
	torch.save(corr, args.outf + '/%s_%d_grc.pth' %(args.corruption, args.level))
