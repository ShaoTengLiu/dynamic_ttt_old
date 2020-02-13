import numpy as np
import torch
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette('colorblind')

corruptions_names = ['original', 'gauss', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 
						'snow', 'frost', 'fog', 'bright', 'contrast', 'elastic', 'pixelate', 'jpeg']
corruptions_names.append('scale_16')

corruptions_names_short = ['orig', 'gauss', 'shot', 'impul', 'defoc', 'glass', 'motn', 'zoom', 
						'snow', 'frost', 'fog', 'brit', 'contr', 'elast', 'pixel', 'jpeg']
corruptions_names_short.append('scale')

info = []
info.append(('gn', '_expand_final', 1))
info.append(('gn', '_expand_final', 2))
info.append(('gn', '_expand_final', 3))
info.append(('gn', '_expand_final', 4))
info.append(('gn', '_expand_final', 5))
info.append(('bn', '_expand_final', 5))
# info.append(('gn', '_expand', 5))
# info.append(('gn', '_expand', 4))
# info.append(('gn', '_expand', 3))
# info.append(('gn', '_expand', 2))
# info.append(('gn', '_expand', 1))
# info.append(('bn', '_expand', 5))
# I changed this part, which was wrong

########################################################################

def easy_barplot(table, fname, width=0.2):
	labels = ['Baseline', 'Joint training', 'Test-time training', 'Test-time training online']
	index =  np.asarray(range(len(table[0,:])))

	plt.figure(figsize=(9, 2.5))
	for i, row in enumerate(table):
		plt.bar(index + i*width, row, width, label=labels[i])

	plt.ylabel('Error (%)')
	plt.xticks(index + width/4, corruptions_names)
	plt.xticks(rotation=45)
	plt.legend(prop={'size': 8})
	plt.tight_layout(pad=0)
	plt.savefig(fname)
	plt.close()

def easy_latex(table, prec1=True):
	for row in table:
		row_str = ''
		for entry in row:
			if prec1:
				row_str += '& %.1f' %(entry)
			else:
				row_str += '& %s' %(entry)
		print(row_str)

def table_transform(table_1, table, count):
	for i in range(4):
		print(table[i])
		table[i][16] = table_1[i][16 + count]
	return table

count = 1
for parta, partb, level in info:
	print(level, parta + partb)
	results = torch.load('results/C10C_layer2_%d_%s%s.pth' %(level, parta, partb))

	if parta == 'bn':
		results = results[0:3,:]
	if count == 1:
		table_1 = results
	if parta == 'gn': #### trick
		results = table_transform(table_1, results, count)
	easy_barplot(results, 'results/C10C_layer2_%d_%s%s.pdf' %(level, parta, partb))
	easy_latex([corruptions_names_short], prec1=False)
	easy_latex(results)
	count += 1
