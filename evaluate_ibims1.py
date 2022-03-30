import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
from utils import *
from models.net import DANet
from options import get_args
from dataloader import dataloader
from evaluate_ibims_error_metrics import compute_global_errors,\
                                         compute_directed_depth_error,\
                                         compute_depth_boundary_error,\
                                         compute_planarity_error,\
                                         compute_distance_related_errors

args = get_args('test')
model = DANet(args)
model = torch.nn.DataParallel(model)
model.cuda()
with open('./data/iBims1/imagelist.txt') as f:
    image_names = f.readlines()
image_names = [x.strip() for x in image_names]
num_samples = len(image_names) # number of images
# Initialize global and geometric errors ...
rms     = np.zeros(num_samples, np.float32)
log10   = np.zeros(num_samples, np.float32)
abs_rel = np.zeros(num_samples, np.float32)
sq_rel  = np.zeros(num_samples, np.float32)
thr1    = np.zeros(num_samples, np.float32)
thr2    = np.zeros(num_samples, np.float32)
thr3    = np.zeros(num_samples, np.float32)

abs_rel_vec = np.zeros((num_samples,20),np.float32)
log10_vec = np.zeros((num_samples,20),np.float32)
rms_vec = np.zeros((num_samples,20),np.float32)

dde_0   = np.zeros(num_samples, np.float32)
dde_m   = np.zeros(num_samples, np.float32)
dde_p   = np.zeros(num_samples, np.float32)

dbe_acc = np.zeros(num_samples, np.float32)
dbe_com = np.zeros(num_samples, np.float32)

pe_fla = np.empty(0)
pe_ori = np.empty(0)

TestImgLoader = dataloader.getTestingData_iBims1(1)

# load test model
if args.loadckpt is not None:
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)["model"]
    model.load_state_dict(state_dict)
else:
    print("You have not loaded any models.")

def test():
    model.eval()

    for i, sample_batched in tqdm(enumerate(TestImgLoader),total=len(TestImgLoader)):
        print("Processing the {}th image!".format(i))
        input, depth, edges, calib, mask_invalid, mask_transp, mask_wall, \
        paras_wall, mask_table, paras_table, mask_floor, paras_floor=sample_batched['image'], sample_batched['depth'], \
                                                                     sample_batched['edges'], sample_batched['calib'], \
                                                                     sample_batched['mask_invalid'], sample_batched[
                                                                         'mask_transp'], \
                                                                     sample_batched['mask_wall'], sample_batched[
                                                                         'mask_wall_paras'], \
                                                                     sample_batched['mask_table'], sample_batched[
                                                                         'mask_table_paras'], \
                                                                     sample_batched['mask_floor'], sample_batched[
                                                                         'mask_floor_paras']

        input=input.cuda()
        bin, pred=model(input)
        pred=pred[-1]
        pred=torch.nn.functional.interpolate(pred, size=[depth.size(2), depth.size(3)], mode='nearest')

        pred=pred.detach().cpu().numpy().squeeze()
        depth=depth.detach().cpu().numpy().squeeze()
        edges=edges.numpy().squeeze()
        calib=calib.numpy().squeeze()
        mask_transp=mask_transp.numpy().squeeze()
        mask_invalid=mask_invalid.numpy().squeeze()
        mask_wall=mask_wall.numpy().squeeze()
        paras_wall=paras_wall.numpy().squeeze()
        mask_table=mask_table.numpy().squeeze()
        paras_table=paras_table.numpy().squeeze()
        mask_floor=mask_floor.numpy().squeeze()
        paras_floor=paras_floor.numpy().squeeze()

        pred[np.isnan(pred)]=0
        pred_invalid=pred.copy()
        pred_invalid[pred_invalid!=0]=1

        mask_missing=depth.copy()  # Mask for further missing depth values in depth map
        mask_missing[mask_missing!=0]=1

        mask_valid=mask_transp*mask_invalid*mask_missing*pred_invalid  # Combine masks
        # Apply 'valid_mask' to raw depth map
        depth_valid=depth*mask_valid

        gt=depth_valid
        gt_vec=gt.flatten()

        # Apply 'valid_mask' to raw depth map
        pred=pred*mask_valid
        pred_vec=pred.flatten()
        # Compute errors ...
        abs_rel[i], sq_rel[i], rms[i], log10[i], thr1[i], thr2[i], thr3[i]=compute_global_errors(gt_vec, pred_vec)
        abs_rel_vec[i, :], log10_vec[i, :], rms_vec[i, :]=compute_distance_related_errors(gt, pred)
        dde_0[i], dde_m[i], dde_p[i]=compute_directed_depth_error(gt_vec, pred_vec, 3.0)
        dbe_acc[i], dbe_com[i], est_edges=compute_depth_boundary_error(edges, pred)

        ##scipy.misc.imsave(image_name+'_predictions_planenet_edges.png', est_edges.astype(int))

        mask_wall=mask_wall*mask_valid
        global pe_fla, pe_ori

        if paras_wall.size>0:
            pe_fla_wall, pe_ori_wall=compute_planarity_error(gt, pred, paras_wall, mask_wall, calib)
            pe_fla=np.append(pe_fla, pe_fla_wall)
            pe_ori=np.append(pe_ori, pe_ori_wall)

        mask_table=mask_table*mask_valid
        if paras_table.size>0:
            pe_fla_table, pe_ori_table=compute_planarity_error(gt, pred, paras_table, mask_table, calib)
            pe_fla=np.append(pe_fla, pe_fla_table)
            pe_ori=np.append(pe_ori, pe_ori_table)

        mask_floor=mask_floor*mask_valid
        if paras_floor.size>0:
            pe_fla_floor, pe_ori_floor=compute_planarity_error(gt, pred, paras_floor, mask_floor, calib)
            pe_fla=np.append(pe_fla, pe_fla_floor)
            pe_ori=np.append(pe_ori, pe_ori_floor)

    print('Results:')
    print('############ Global Error Metrics #################')
    print('rel    = ', np.nanmean(abs_rel))
    print('sq_rel = ', np.nanmean(sq_rel))
    print('log10  = ', np.nanmean(log10))
    print('rms    = ', np.nanmean(rms))
    print('thr1   = ', np.nanmean(thr1))
    print('thr2   = ', np.nanmean(thr2))
    print('thr3   = ', np.nanmean(thr3))
    print('############ Planarity Error Metrics #################')
    print('pe_fla = ', np.nanmean(pe_fla))
    print('pe_ori = ', np.nanmean(pe_ori))
    print('############ Depth Boundary Error Metrics #################')
    print('dbe_acc = ', np.nanmean(dbe_acc))
    print('dbe_com = ', np.nanmean(dbe_com))
    print('############ Directed Depth Error Metrics #################')
    print('dde_0  = ', np.nanmean(dde_0)*100.)
    print('dde_m  = ', np.nanmean(dde_m)*100.)
    print('dde_p  = ', np.nanmean(dde_p)*100.)

if __name__ == '__main__':
    test()
