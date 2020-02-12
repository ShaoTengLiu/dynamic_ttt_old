import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize(*NORM)]) # 转换图片 version
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(*NORM)])
mnist_transforms = transforms.Compose([transforms.Resize((32, 32)),
										transforms.ToTensor(),
										transforms.Normalize((0.1307,), (0.3081,))])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
for i in range(1, 9):
	common_corruptions.append('scale' + str(i))

def prepare_test_data(args): # 这个是加入scale的关键
	if args.dataset == 'cifar10':
		tesize = 10000
		if not hasattr(args, 'corruption') or args.corruption == 'original':
			print('Test on the original test set')
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms) # 如果没有corupption，就用原来的test
		elif args.corruption in common_corruptions:
			print('Test on %s level %d' %(args.corruption, args.level))
			teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption)) # 所以最后要得到这么个API
			teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize] # 剪裁出对应等级的一部分
			teset = torchvision.datasets.CIFAR10(root=args.dataroot,
												train=False, download=True, transform=te_transforms) # 这里只是获得训练数据
			teset.data = teset_raw # 这一步改变了测试集

		elif args.corruption == 'cifar_new':
			from utils.cifar_new import CIFAR_New
			print('Test on CIFAR-10.1')
			teset = CIFAR_New(root=args.dataroot + 'CIFAR-10.1/datasets/', transform=te_transforms)
			permute = False
		else:
			raise Exception('Corruption not found!')
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
											shuffle=False, num_workers=args.workers) # loader是由teset产生的
	return teset, teloader # 注意返回的东西，感受数据集 loader的使用方法

def prepare_train_data(args):
	print('Preparing data...')
	if args.dataset == 'cifar10':
		trset = torchvision.datasets.CIFAR10(root=args.dataroot,
										train=True, download=True, transform=tr_transforms)
	else:
		raise Exception('Dataset not found!')

	if not hasattr(args, 'workers'):
		args.workers = 1
	trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
											shuffle=True, num_workers=args.workers)
	return trset, trloader
	