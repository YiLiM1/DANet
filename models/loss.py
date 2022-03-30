import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss import chamfer_distance

class Sobel(nn.Module):
	def __init__(self):
		super(Sobel, self).__init__()
		self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
		edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
		edge_k = np.stack((edge_kx, edge_ky))

		edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
		self.edge_conv.weight = nn.Parameter(edge_k)
		
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		out = self.edge_conv(x) 
		out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
		return out

class Total_loss(nn.Module):
	def __init__(self):
		super().__init__()
		self.name='L1'
	def forward(self,output, depth_gt):
		losses=[]
		for depth_index in range(len(output)):
			output[depth_index]=F.interpolate(output[depth_index], size=depth_gt[depth_index].size()[2:4], mode='bilinear',align_corners=True)
			cos = nn.CosineSimilarity(dim=1, eps=0)
			get_gradient = Sobel().cuda()
			ones = torch.ones(depth_gt[depth_index].size(0), 1, depth_gt[depth_index].size(2),depth_gt[depth_index].size(3)).float().cuda()
			ones = torch.autograd.Variable(ones)
			depth_grad = get_gradient(depth_gt[depth_index])
			output_grad = get_gradient(output[depth_index])
			depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth_gt[depth_index])
			depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth_gt[depth_index])
			output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth_gt[depth_index])
			output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth_gt[depth_index])

			depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
			output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

			# loss_depth=torch.abs(output[depth_index]-depth_gt[depth_index]).mean()
			# loss_dx=torch.abs(output_grad_dx-depth_grad_dx).mean()
			# loss_dy=torch.abs(output_grad_dy-depth_grad_dy).mean()
			loss_depth = torch.log(torch.abs(output[depth_index] - depth_gt[depth_index]) + 0.5).mean()
			loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
			loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
			loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
			loss=loss_depth+loss_normal+(loss_dx+loss_dy)
			losses.append(loss)
		total_loss = sum(losses)
		return total_loss

class MinmaxLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.name = "MinMaxLoss"
	def forward(self, bins, target_depth_maps):
		# print(target_depth_maps.shape)
		# print(bins[:,1])
		# print(target_depth_maps.min(dim=2)[0].min(dim=2)[0].squeeze(dim=1))
		bin_centers=0.5*(bins[:, 1:]+bins[:, :-1])
		losses=[]
		for i in range(target_depth_maps.shape[0]):
			gt=target_depth_maps[i][target_depth_maps[i]>0.01]
			try:
				max_loss=(bin_centers[i,-1]-gt.max()).abs()
				min_loss=(bin_centers[i, 0]-gt.min()).abs()
			except:
				# plt.imshow(target_depth_maps[i].squeeze().cpu())
				# plt.show()
				pass

			loss=max_loss+min_loss
			losses.append(loss)
		total_loss=sum(losses)

		#print(max_loss.data,min_loss.data)
		return total_loss/target_depth_maps.shape[0]

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
	def __init__(self,alpha=10):
		super(SILogLoss, self).__init__()
		self.name = 'SILog'
		self.alpha=alpha

	def forward(self, output,depth_gt):
		losses=[]
		for i in range(len(output)):
			mask_pred = output[i]>0.01
			mask_gt=depth_gt[i]>0.01
			mask=torch.logical_and(mask_pred,mask_gt)
			input = output[i][mask]
			target = depth_gt[i][mask]
			# mask=depth_gt[i]>0.01
			# input = output[i][mask]
			# target = depth_gt[i][mask]
			g = torch.log(input+0.1) - torch.log(target+0.1)
			# n, c, h, w = g.shape
			# norm = 1/(h*w)
			# Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2
			Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
			losses.append(self.alpha * torch.sqrt(Dg))
		total_loss=sum(losses)
		return total_loss

class SILogLoss_valley(nn.Module):  # Main loss function used in AdaBins paper
	def __init__(self,factor=2,alpha=10):
		super(SILogLoss_valley, self).__init__()
		self.name = 'SILog_valley'
		self.factor=factor
		self.alpha=alpha

	def forward(self, output,depth_gt):
		losses=[]
		for i in range(len(output)):
			mask_pred = output[i]>0.01
			mask_gt=depth_gt[i]>0.01
			mask=torch.logical_and(mask_pred,mask_gt)
			input = output[i][mask]
			target = depth_gt[i][mask]
			# mask=depth_gt[i]>0.01
			# input=output[i][mask]
			# target=depth_gt[i][mask]
			copy=target.clone().detach()

			median=copy.median()
			#median=1.8
			wavelength_large=1/((copy.max()-median))
			wavelength_small=1/((median-copy.min()))
			weight=copy-median
			mask=weight<0
			weight=(copy-median).abs()
			weight=wavelength_large*weight
			weight[mask]*=wavelength_small/wavelength_large
			weight=weight*(self.factor-1)
			weight=weight+1
			g = weight*(torch.log(input+0.1) - torch.log(target+0.1))


			# plt.imshow(weight.cpu().numpy().squeeze())
			# plt.show()
			# plt.imshow(target.cpu().numpy().squeeze())
			# plt.show()
			# n, c, h, w = g.shape
			# norm = 1/(h*w)
			# Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2
			Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
			losses.append(self.alpha * torch.sqrt(Dg))
		total_loss=sum(losses)
		return total_loss

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape
        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1
        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
