import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import torch
import time
import math
import cv2
import os

from models.yolov6 import non_max_suppression
from models.utils.yolo_utils import scale_coords, xywh2xyxy , box_iou , xyxy2xywh 

def time_synchronized(use_cpu=False):
    torch.cuda.synchronize(
    ) if torch.cuda.is_available() and not use_cpu else None
    return time.time()

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0],
                          iouv.shape[0],
                          dtype=torch.bool,
                          device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # IoU above threshold and classes match
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                            1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def get_batch_statistics(imgs, targets, paths, shapes0, output, seen, stats, json_dict):
    """ Compute true positives, predicted scores and predicted labels per sample """
    device = targets.device
    
    _, _, height, width = imgs.shape  # batch size, channels, height, width
    targets[:, 2:] *= torch.Tensor([width, height, width,
                                    height]).to(device)  # to pixels
    # iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()  # number of iou (10)

    # Statistics per image
    for si, pred in enumerate(output):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        path, shape0 = Path(paths[si]), shapes0[si]
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool),
                              torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        predn[:, :4] = scale_coords(imgs[si].shape[1:], predn[:, :4],
                                    shape0)  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            tbox = scale_coords(imgs[si].shape[1:], tbox,
                                shape0)  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox),
                                1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(),
                      tcls))  # (correct, conf, pcls, tcls)

    return seen, stats, json_dict

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(img, c1, c2, color, thickness=tl)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def plot_images(images,targets,paths=None,fname='images.jpg',names=None,max_size=640,max_subplots=16,overwrite=False):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if not overwrite:
        if os.path.isfile(fname):  # do not overwrite
            return None

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:,
                                                 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box,
                                 mosaic,
                                 label=label,
                                 color=color,
                                 line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3,
                                     thickness=tf)[0]
            cv2.putText(mosaic,
                        label, (block_x + 5, block_y + t_size[1] + 5),
                        0,
                        tl / 3, [220, 220, 220],
                        thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h),
                      (255, 255, 255),
                      thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)),
                            interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic

def output_to_target(output, shape):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    width, height = shape
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            box = np.array(box) / [width, height, width, height]
            targets.append(
                [i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)

def plot_pr_curve(px, py, ap, save_dir='.', names=()):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # show mAP in legend if < 10 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} %.3f' %
                    ap[i, 0])  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px,
            py.mean(1),
            linewidth=3,
            color='blue',
            label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def plot_mc_curve(px,py,save_dir='mc_curve.png',names=(),xlabel='Confidence',ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1,
                    label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1,
                color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px,
            y,
            linewidth=3,
            color='blue',
            label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(
            mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def ap_per_class(tp,conf,pred_cls,target_cls,plot=False,save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros(
        (nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0],
                              left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                              left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:,
                                                                           j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec,
                                        mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px,
                      f1,
                      Path(save_dir) / 'F1_curve.png',
                      names,
                      ylabel='F1')
        plot_mc_curve(px,
                      p,
                      Path(save_dir) / 'P_curve.png',
                      names,
                      ylabel='Precision')
        plot_mc_curve(px,
                      r,
                      Path(save_dir) / 'R_curve.png',
                      names,
                      ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    # i = r.mean(0).argmax()  # max Recall index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

def print_stats_to_console(seen,nt,mp,mr,map50,print_stats,class_names,num_classes,stats,ap_class,p,r,ap50,ap,t0,t1,t2,batch_size,img_size,run_dir):
    #open the results text file if it exists 
    if not os.path.exists(run_dir + '/results.txt'):
        with open(run_dir + '/results.txt', 'x') as f:
            f.write('')
            
    with open(run_dir + '/results.txt', 'a') as f:

        # Print results
        pf = '%20s' + '%12.4g' * 6  # print format
        if print_stats:
            print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        
        f.write(pf % ('all', seen, nt.sum(), mp, mr, map50, map) + '\n')

        # Print results per class
        if num_classes > 1 and num_classes <= 20 and len(stats):
            for i, c in enumerate(ap_class):
                if print_stats:
                    print(pf %(class_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                
                f.write(pf % (class_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')

        # Print speeds
        times = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
        shape = (batch_size, 3, img_size[0], img_size[1])
        if print_stats:
            print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}'% times)
        
        f.write(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}'% times + '\n\n')

@torch.no_grad()
def evaluate(model,dataloader,class_names , img_size ,device,conf_thres = 0.001,nms_thres = 0.5,run_dir = './runs',verbose=False , print_stats = False):
    '''
    @param model: model to evaluate
    @param data_loader: data loader for the dataset to evaluate
    @param class_names: list of class names
    @param img_size: size of each image dimension
    @param device: device to run the model on
    @param conf_thres: confidence threshold for NMS
    @param nms_thres: NMS threshold
    @param run_dir: directory to create a folder in and save the results of the evaluation
    @param verbose: whether to print stats
    '''

    if isinstance(img_size, int):
        img_size = (img_size, img_size)


    #initialize variables
    num_classes = len(class_names)
    loss = torch.zeros(3 , device=device)
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R','mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    seen = 0
    json_dict, stats, ap, ap_class = [], [], [], []

    #Create a folder to save the results of the evaluation
    f = Path(run_dir)
    if not os.path.exists(f):
        os.makedirs(f)   

    #set model to evaluation mode
    model.eval()

    model(torch.zeros(1, 3, img_size[1], img_size[0]).to(device).type_as(next(model.parameters())))

    for batch_i, (imgs, targets, paths,  shapes0) in enumerate(tqdm(dataloader , desc="Evaluating")):
        t = time_synchronized()
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.float()  # uint8 to fp16/32

        targets = targets.to(device)

        t0 += time_synchronized() - t


        # Run model
        t = time_synchronized()
        pred , _ = model(imgs)  # inference and training outputs
        t1 += time_synchronized() - t
        
        # Run NMS
        t = time_synchronized()
        output = non_max_suppression(pred, conf_thres, nms_thres)
        t2 += time_synchronized() - t

        # Statistics per batch
        seen, stats, json_dict = get_batch_statistics(imgs, targets.clone(),paths, shapes0, output,seen, stats,json_dict)
        
        if batch_i < 3 and verbose:         
            f = Path(run_dir) / ('test_batch%g_gt.jpg' % batch_i)
            # plot ground truth
            plot_images(imgs, targets, paths, str(f), class_names)
            f = Path(run_dir) / ('test_batch%g_pred.jpg' % batch_i)
            # plot predictions
            plot_images(imgs, output_to_target(output, imgs.shape[2:4]), paths,str(f), class_names)


    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        print("STATS FOUND")
        p, r, ap, f1, ap_class = ap_per_class(*stats,plot=verbose,save_dir=run_dir,names=class_names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # number of targets per class
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_classes)
    else:
        print("NO STATS FOUND")
        nt = torch.zeros(1)

    # Print batch results

    print_stats_to_console(seen,nt,mp,mr,map50,print_stats,class_names,num_classes,stats,ap_class,p,r,ap50,ap,t0,t1,t2,dataloader.batch_size,img_size , run_dir)

    return (mp, mr, map50, map,*(loss.cpu() / len(dataloader)).tolist())

