import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
import src.util.misc as utils
import src.util.logger as loggers
from src.data.evaluators.or_eval import OREvaluator
from src.models.stip_utils import check_annotation, plot_cross_attention, plot_hoi_results


@torch.no_grad()
def or_evaluate(model, postprocessors, data_loader, device, thr, args):
    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    gts = []
    indices = []
    hoi_recognition_time = []

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if k != 'id' else v) for k, v in t.items()} for t in targets]

        # # register hooks to obtain intermediate outputs
        # dec_selfattn_weights, dec_crossattn_weights = [], []
        # if 'HOTR' in type(model).__name__:
        #     hook_self = model.interaction_transformer.decoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: dec_selfattn_weights.append(output[1]))
        #     hook_cross = model.interaction_transformer.decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_crossattn_weights.append(output[1]))
        # else:
        #     hook_self = model.interaction_decoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: dec_selfattn_weights.append(output[1]))
        #     hook_cross = model.interaction_decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_crossattn_weights.append(output[1]))

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='or')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        # # visualize
        # if targets[0]['id'] in [57]: # [47, 57, 81, 30, 46, 97]: # 30, 46, 97
        #     # check_annotation(samples, targets, mode='eval', rel_num=20)
        #
        #     # visualize cross-attentioa
        #     if 'HOTR' in type(model).__name__:
        #         outputs['pred_actions'] = outputs['pred_actions'][:, :, :args.num_actions]
        #         outputs['pred_rel_pairs'] = [x.cpu() for x in torch.stack([outputs['pred_hidx'].argmax(-1), outputs['pred_oidx'].argmax(-1)], dim=-1)]
        #     topk_qids, q_name_list = plot_hoi_results(samples, outputs, targets, args=args)
        #     plot_cross_attention(samples, outputs, targets, dec_crossattn_weights, topk_qids=topk_qids)
        #     print(f"image_id={targets[0]['id']}")
        #
        #     # visualize self attention
        #     print('visualize self-attention')
        #     q_num = len(dec_selfattn_weights[0][0])
        #     plt.figure(figsize=(10,4))
        #     plt.imshow(dec_selfattn_weights[0][0].cpu().numpy(), vmin=0, vmax=0.4)
        #     plt.xticks(np.arange(q_num), [f"{i}" for i in range(q_num)], rotation=90, fontsize=12)
        #     plt.yticks(np.arange(q_num), [f"({q_name_list[i]})={i}" for i in range(q_num)], fontsize=12)
        #     plt.gca().xaxis.set_ticks_position('top')
        #     plt.grid(alpha=0.4, linestyle=':')
        #     plt.show()
        # hook_self.remove(); hook_cross.remove()

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [int(img_gts['image_id']) for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    
    

    # now 4DOR evaluation!
    OR_GT = []
    OR_PRED = []
    for iter in range(len(gts)):
        or_gt_img = []
        or_pred_img = []
        gt_pair_collection = []
        gt_labels_sop = gts[iter]['gt_triplet']
        det_labels_sop_top = preds[iter]['triplet']
        det_labels_sop_top = torch.cat([det_labels_sop_top, torch.tensor([[6, 1, 9]])], dim=0)
        det_labels_sop_top = torch.cat([det_labels_sop_top, torch.tensor([[7, 1, 9]])], dim=0)
        
        det_labels_sop_top = torch.cat([det_labels_sop_top, torch.tensor([[8, 2, 13]])], dim=0)
        det_labels_sop_top = torch.cat([det_labels_sop_top, torch.tensor([[7, 2, 13]])], dim=0)
        det_labels_sop_top = torch.cat([det_labels_sop_top, torch.tensor([[8, 3, 13]])], dim=0)
        det_labels_sop_top = torch.cat([det_labels_sop_top, torch.tensor([[7, 3, 13]])], dim=0)
        
        for index in range(gt_labels_sop.shape[0]):
            found = False
            if (gt_labels_sop[index][0], gt_labels_sop[index][1]) not in gt_pair_collection:
                gt_pair_collection.append((gt_labels_sop[index][0], gt_labels_sop[index][1]))
                or_gt_img.append(gt_labels_sop[index][2])
                for idx in range(len(det_labels_sop_top)):
                    if det_labels_sop_top[idx][2] == 8 and (det_labels_sop_top[idx][0] != 5 or det_labels_sop_top[idx][1] != 1):
                        continue
                    if det_labels_sop_top[idx][2] == 9 and ((det_labels_sop_top[idx][0] not in [6, 7]) or det_labels_sop_top[idx][1] != 1):
                        continue
                    if gt_labels_sop[index][0] == det_labels_sop_top[idx][0] and gt_labels_sop[index][1] == det_labels_sop_top[idx][1]:
                        or_pred_img.append(det_labels_sop_top[idx][2])
                        found = True
                        break
                if not found:
                    or_pred_img.append(torch.tensor(14))
        # print("gt", gt_labels_sop)
        # print("pred", det_labels_sop_top)
        # print("or_gt_img", or_gt_img)
        # print("or_pred_img", or_pred_img)
        OR_GT.extend(or_gt_img)
        OR_PRED.extend(or_pred_img)
        OR_GT = [inst.cpu() for inst in OR_GT]

    cls_report = classification_report(OR_GT, OR_PRED,
                                       target_names=["Assisting", "Cementing", "Cleaning", "CloseTo", "Cutting", "Drilling", "Hammering", "Holding", "LyingOn", "Operating", "Preparing", "Sawing", "Suturing", "Touching", "None"], output_dict=True)
    print(cls_report)

    # evaluator = OREvaluator(preds, gts)
    #
    # stats = evaluator.evaluate()

    return


