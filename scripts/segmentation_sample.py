"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
# viz = Visdom(port=8850)
viz = Visdom()
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from medpy import metric
import echonet
import numpy as np
import torch
from guided_diffusion.myresnet50 import featurepreprocess


seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=args.data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }

    dataset = {}
    dataset["test"] = echonet.datasets.Echo(root=args.data_dir, split="test", **kwargs)
    print(len(dataset["test"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = dataset["test"]
    datal = torch.utils.data.DataLoader(
                    ds, batch_size=args.batch_size, num_workers=4, shuffle=False)
    file_name=datal.dataset.fnames
    print(file_name)
    print(len(file_name))
    ffnames=iter(file_name)
    data = iter(datal)

    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    output = os.path.join("./results", "Dice")
    
    os.makedirs(output, exist_ok=True)
    
    with open(os.path.join(output, "log.csv"), "a") as f:
        large_inter_list = []
        large_union_list = []
        small_inter_list = []
        small_union_list = []

        total_metric_hd=0
        large_metric_hd=0
        small_metric_hd=0
        
        for i in range(len(datal)):
            logger.log("Starting run data {}".format(i))
            f.flush()
            a, d = next(data)  #should return an image from the dataloader "data"
            IDname=next(ffnames)
            logger.log("IDname: {}".format(IDname))
            save_name=os.path.splitext(IDname)[0]
            (small_frame, large_frame, small_trace, large_trace)=d
            large_c = th.randn_like(large_frame[:, :1, ...])
            small_c = th.randn_like(small_frame[:, :1, ...])

            img_large = th.cat((large_frame, large_c), dim=1)     #add a noise channel$
            img_small = th.cat((small_frame, small_c), dim=1)     #add a noise channel$
            # slice_ID=path[0].split("/", -1)[3]
            large_mask=large_trace
            small_mask=small_trace

            print("===================================")

            #这一部分处理 Large_frame 帧
            viz.image(visualize(img_large[0,:-1,...]), opts=dict(caption="image"))
            viz.image(visualize(img_large[0,3, ...]), opts=dict(caption="L-input_noise"))
            viz.image(visualize(large_mask), opts=dict(caption="largemask"))
            logger.log("sampling large_frame...")

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)

            for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
               
                sample, x_noisy, org = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size), img_large,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                logger.log("++++++++++++++++++++++++++")
                logger.log((args.batch_size, 3, args.image_size, args.image_size))
                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

                s = th.tensor(sample)
                th.save(s, './results/sampled/'+'L-'+str(save_name)+ '_output'+str(i)) #save the generated mask
                
                viz.image(visualize(large_mask), opts=dict(caption="L-img_mask"))
                viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled_output"))

                pred=s
                pred=pred.to("cpu")
                pred=visualize(pred)
                print("end============================")

                gt=large_trace

                large_inter,large_union=compute(gt,pred)
                large_dice=2 * large_inter / (large_union + large_inter)
                   
                large_inter_list.extend(large_inter)
                large_union_list.extend(large_union)

                large_hd=calculate_metric_percase(pred,gt)
                large_metric_hd += np.asarray(large_hd)

                logger.log("num_ensemble:{}, name:{}, large_dice:{},large_hd:{}\n".format(i,str(save_name),large_dice,large_hd)) 
                

                
            
            #这一部分处理 Small_frame 帧
            viz.image(visualize(img_small[0,:-1,...]), opts=dict(caption="image"))
            viz.image(visualize(img_small[0,3, ...]), opts=dict(caption="S-input_noise"))
            viz.image(visualize(small_mask), opts=dict(caption="smallmask"))
            logger.log("sampling small_frame...")

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)

            for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
               
                sample, x_noisy, org = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size), img_small,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

                s = th.tensor(sample)
                th.save(s, './results/sampled/'+'S-'+str(save_name)+ '_output'+str(i)) #save the generated mask
                
                viz.image(visualize(small_mask), opts=dict(caption="S-img_mask"))
                viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled_output"))

                pred=s
                pred=pred.to("cpu")

                pred=visualize(pred)
                print("end============================")

                gt=small_trace
                
                small_inter,small_union=compute(gt,pred)
                small_dice=2 * small_inter / (small_union + small_inter)
                   
                small_inter_list.extend(small_inter)
                small_union_list.extend(small_union)

                small_hd=calculate_metric_percase(pred,gt)
                small_metric_hd += np.asarray(small_hd)

                logger.log("num_ensemble:{}, name:{}, small_dice:{},small_hd:{}\n".format(i,str(save_name),small_dice,small_hd)) 
                
                
        large_inter_list = np.array(large_inter_list)
        large_union_list = np.array(large_union_list)
        small_inter_list = np.array(small_inter_list)
        small_union_list = np.array(small_union_list)
        
        total_metric_hd=large_metric_hd+small_metric_hd
        total_aver_hd=total_metric_hd/ (len(file_name)*2)
   
        large_aver_hd=large_metric_hd/len(file_name)
        small_aver_hd=small_metric_hd/len(file_name)
 
        logger.log("dice (overall_dice):{:.4f} ({:.4f} - {:.4f})".format(*echonet.utils.bootstrap(np.concatenate((large_inter_list, small_inter_list)), np.concatenate((large_union_list, small_union_list)), echonet.utils.dice_similarity_coefficient)))
        logger.log("dice (large_dice):{:.4f} ({:.4f} - {:.4f})".format(*echonet.utils.bootstrap(large_inter_list, large_union_list, echonet.utils.dice_similarity_coefficient)))
        logger.log("dice (small_dice):{:.4f} ({:.4f} - {:.4f})".format(*echonet.utils.bootstrap(small_inter_list, small_union_list, echonet.utils.dice_similarity_coefficient)))
        
        
        logger.log("hd (overall_hd):{}\n".format(total_aver_hd))
        logger.log("hd (large_hd): hd:{}\n".format(large_aver_hd))
        logger.log("hd (small_hd): hd:{}\n".format(small_aver_hd))
    

def compute(large_trace,pred):
    total = 0.
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    # large_inter_list = []
    # large_union_list = []
    # small_inter_list = []
    # small_union_list = []

    large_inter = np.logical_and(pred[:, 0, :, :].detach().cpu().numpy() > 0.5, large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1,2))
    large_union = np.logical_or(pred[:, 0, :, :].detach().cpu().numpy() > 0.5, large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1,2))
    
    logger.log("预测像素个数")
    # logger.log(pred[:,0,:,:])
    logger.log((pred[:, 0, :, :].detach().cpu().numpy() > 0.5).sum((1,2)))
    # logger.log(large_trace)
    logger.log("标签像素个数")
    logger.log((large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1,2)))
    

    return (large_inter,
            large_union,
            # small_inter_list,
            # small_union_list,
            ) 

def calculate_metric_percase(pred, gt):
    pred=(pred>0.1).type(torch.float32)
    pred= pred.squeeze(dim=0)
    pred=pred.numpy()
    gt=gt.numpy()
    hd = metric.binary.hd(pred, gt)
    return hd


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
