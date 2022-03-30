import os
import math
import torch
import numpy as np
def makedir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def adjust_gt(gt_depth, pred_depth):
	adjusted_gt = []
	for each_depth in pred_depth:
		adjusted_gt.append(torch.nn.functional.interpolate(gt_depth, size=[each_depth.size(2), each_depth.size(3)],
								   mode='bilinear', align_corners=True))
	return adjusted_gt

def adjust_learning_rate(optimizer, epoch, init_lr,stage=5):
	lr = init_lr * (0.1 ** (epoch // stage))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def save_checkpoint(state, filename):
	torch.save(state, filename)

def build_optimizer(model,
					learning_rate, 
					optimizer_name='rmsprop',
					weight_decay=1e-5,
					epsilon=0.001,
					momentum=0.9):
	"""Build optimizer"""
	if optimizer_name == "sgd":
		print("Using SGD optimizer.")
		optimizer = torch.optim.SGD(model.parameters(), 
									lr = learning_rate,
									momentum=momentum,
									weight_decay=weight_decay)

	elif optimizer_name	== 'rmsprop':
		print("Using RMSProp optimizer.")
		optimizer = torch.optim.RMSprop(model.parameters(),
										lr = learning_rate,
										eps = epsilon,
										weight_decay = weight_decay,
										momentum = momentum
										)
	elif optimizer_name == 'adam':
		print("Using Adam optimizer.")

		Encoder=list(map(id, model.module.feature_extraction.parameters()))
		base_params=filter(lambda p: id(p) not in Encoder, model.module.parameters())
		optimizer = torch.optim.Adam([{'params': base_params},
									  {'params': model.module.feature_extraction.parameters(), 'lr':learning_rate*0.1}],
									 lr = learning_rate, weight_decay=weight_decay)

	return optimizer

def lg10(x):
	return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
	z = x.clone()
	maskYLarger = torch.lt(x, y)
	z[maskYLarger.detach()] = y[maskYLarger.detach()]
	return z

def nValid(x):
	return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
	return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
	return torch.ne(x, x)

def setNanToZero(input, target):
	nanMask = getNanMask(target)
	nValidElement = nValid(target)

	_input = input.clone()
	_target = target.clone()

	_input[nanMask] = 0
	_target[nanMask] = 0

	return _input, _target, nanMask, nValidElement

def evaluateError(output, target):
	errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
			  'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0,'RMSE_FD': 0}

	mask=target>0.01
	output=output[mask]
	target=target[mask]
	_output, _target, nanMask, nValidElement = setNanToZero(output, target)

	if (nValidElement.data.cpu().numpy() > 0):
		diffMatrix = torch.abs(_output -_target)

		errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement
		errors['RMSE_FD']=errors['MSE']**0.5
		errors['MAE'] = torch.sum(diffMatrix) / nValidElement

		realMatrix = torch.div(diffMatrix, _target)
		realMatrix[nanMask] = 0
		errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

		LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
		LG10Matrix[nanMask] = 0
		errors['LG10'] = torch.sum(LG10Matrix) / nValidElement
		yOverZ = torch.div(_output, _target)
		zOverY = torch.div(_target, _output)

		maxRatio = maxOfTwo(yOverZ, zOverY)

		errors['DELTA1'] = torch.sum(
			torch.le(maxRatio, 1.25).float()) / nValidElement
		errors['DELTA2'] = torch.sum(
			torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
		errors['DELTA3'] = torch.sum(
			torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

		errors['RMSE_FD']=float(errors['RMSE_FD'].data.cpu().numpy())
		errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
		errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
		errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
		errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
		errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
		errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
		errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

	return errors

def addErrors(errorSum, errors, batchSize):
	errorSum['RMSE_FD']=errorSum['RMSE_FD']+errors['RMSE_FD']*batchSize
	errorSum['MSE']=errorSum['MSE'] + errors['MSE'] * batchSize
	errorSum['ABS_REL']=errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
	errorSum['LG10']=errorSum['LG10'] + errors['LG10'] * batchSize
	errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

	errorSum['DELTA1']=errorSum['DELTA1'] + errors['DELTA1'] * batchSize
	errorSum['DELTA2']=errorSum['DELTA2'] + errors['DELTA2'] * batchSize
	errorSum['DELTA3']=errorSum['DELTA3'] + errors['DELTA3'] * batchSize
	return errorSum

def averageErrors(errorSum, N):
	averageError={'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
					'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0,'RMSE_FD': 0}

	averageError['RMSE_FD']=errorSum['RMSE_FD']/N
	averageError['MSE'] = errorSum['MSE'] / N
	averageError['ABS_REL'] = errorSum['ABS_REL'] / N
	averageError['LG10'] = errorSum['LG10'] / N
	averageError['MAE'] = errorSum['MAE'] / N

	averageError['DELTA1'] = errorSum['DELTA1'] / N
	averageError['DELTA2'] = errorSum['DELTA2'] / N
	averageError['DELTA3'] = errorSum['DELTA3'] / N
	return averageError

class RunningAverage:
	def __init__(self):
		self.avg = 0
		self.count = 0

	def append(self, value):
		self.avg = (value + self.count * self.avg) / (self.count + 1)
		self.count += 1

	def get_value(self):
		return self.avg

class RunningAverageDict:
	def __init__(self):
		self._dict = None

	def update(self, new_dict):
		if self._dict is None:
			self._dict = dict()
			for key, value in new_dict.items():
				self._dict[key] = RunningAverage()

		for key, value in new_dict.items():
			self._dict[key].append(value)

	def get_value(self):
		return {key: value.get_value() for key, value in self._dict.items()}

def compute_errors(gt, pred):
	thresh = np.maximum((gt / pred), (pred / gt))
	a1 = (thresh < 1.25).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	abs_rel = np.mean(np.abs(gt - pred) / gt)
	sq_rel = np.mean(((gt - pred) ** 2) / gt)

	rmse = (gt - pred) ** 2
	rmse = np.sqrt(rmse.mean())

	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	err = np.log(pred) - np.log(gt)
	silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

	log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()


	return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
				silog=silog, sq_rel=sq_rel)

