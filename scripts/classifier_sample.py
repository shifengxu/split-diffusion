"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import sys
from functools import partial
from utils import log_info

import torch as th
import torch.nn.functional as F
import torchvision.utils as tvu

cur_dir = os.path.dirname(__file__) # current dir
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
if prt_dir not in sys.path:
    sys.path.append(prt_dir)

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
)
from config import create_config

def create_model_classifier(args, config, model_config, class_config):
    log_info("create_model_classifier()...")
    model, diffusion = create_model_and_diffusion(**model_config)
    m_path = config['model_path']
    log_info(f"  load diff model: {m_path}")
    s_dict = th.load(m_path, map_location='cpu')
    model.load_state_dict(s_dict)
    log_info(f"  load diff model: {m_path} ... done")
    model.requires_grad_(False).eval()
    model.to(args.device)
    log_info(f"  model.to({args.device})")
    if model_config['use_fp16']:
        model.convert_to_fp16()
        log_info(f"  model.convert_to_fp16()")
    if len(args.gpu_ids) > 1:
        log_info(f"  torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
        model = th.nn.DataParallel(model, device_ids=args.gpu_ids)

    classifier_config = classifier_defaults()
    classifier_config.update(class_config)
    classifier = create_classifier(**classifier_config)
    c_path = config['classifier_path']
    log_info(f"  load classifier: {m_path}")
    s_dict = th.load(c_path, map_location="cpu")
    classifier.load_state_dict(s_dict)
    log_info(f"  load classifier: {m_path} ... done")
    classifier.to(args.device)
    log_info(f"  classifier.to({args.device})")
    classifier.requires_grad_(False).eval()
    if classifier_config['classifier_use_fp16']:
        classifier.convert_to_fp16()
        log_info(f"  classifier.convert_to_fp16()")
    if len(args.gpu_ids) > 1:
        log_info(f"  torch.nn.DataParallel(classifier, device_ids={args.gpu_ids})")
        classifier = th.nn.DataParallel(classifier, device_ids=args.gpu_ids)
    log_info("create_model_classifier()...done")
    return model, diffusion, classifier

def main():
    args, config, model_config, class_config = create_args_config()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"args: {args}")

    model, diffusion, classifier = create_model_classifier(args, config, model_config, class_config)

    def cond_fn(x, t, y=None, **_):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in)[0]
            return grad * config['classifier_scale']

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if model_config['class_cond'] else None)

    if args.method == 'ddim':
        sample_fn = diffusion.ddim_sample_loop
    elif args.method[:4] in ['plms', 'pndm']:
        sample_fn = partial(diffusion.plms_sample_loop, order=int(args.method[4]))
    elif args.method[:4] in ['ltsp', 'ours', 'ltts']:
        sample_fn = partial(diffusion.ltsp_sample_loop, order=int(args.method[4]))
    elif args.method[:4] in ['stsp', 'bchf']:
        sample_fn = partial(diffusion.stsp_sample_loop, order=int(args.method[4]))   
    else:
        sample_fn = diffusion.p_sample_loop 

    if args.cond_name == 'cond1': cond_fn0 = cond_fn
    else: cond_fn0 = None

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    sample_cnt = args.num_samples
    image_size = config['image_size']
    batch_size = config['batch_size']
    batch_cnt = (sample_cnt - 1) // batch_size + 1
    log_info("sampling...")
    log_info(f"sample_cnt : {sample_cnt}")
    log_info(f"batch_size : {batch_size}")
    log_info(f"batch_cnt  : {batch_cnt}")
    log_info(f"image_size : {image_size}")
    log_info(f"out_dir    : {out_dir}")
    for b_idx in range(batch_cnt):
        n = batch_size if b_idx+1 < batch_cnt else sample_cnt - b_idx * batch_size
        classes = th.randint(low=0, high=NUM_CLASSES, size=(n,), device=args.device)
        sample = sample_fn(
            model_fn,
            (n, 3, image_size, image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs={"y": classes},
            cond_fn=cond_fn0,
            impu_fn=None,
            progress=True,
            device=args.device,
        )
        save_samples(sample, classes, out_dir, b_idx, batch_size)
    # while
    log_info("sampling complete")

def save_samples(sample, classes, out_dir, b_idx, b_size):
    sample = ((sample + 1) / 2).clamp(0.0, 1.0)
    img_cnt = len(sample)
    img_path = None
    init_id = b_idx * b_size
    for i in range(img_cnt):
        img_id = init_id + i
        cls_id = classes[i]
        img_path = os.path.join(out_dir, f"{img_id:05d}_c{cls_id:03d}.png")
        tvu.save_image(sample[i], img_path)
    log_info(f"  saved {img_cnt} images: {img_path}")

def create_args_config():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        use_ddim=True,
        model_name="c64",
        method="ddim",
        cond_name="cond1",
        timestep_rp=25,
        output_dir='output1/generated/',
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--todo", type=str, default='alpha_bar_all, schedule_sample')
    # parser.add_argument("--todo", type=str, default='schedule_sample')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7, 6])

    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    log_info(f"gpu_ids : {gpu_ids}")
    args.device = th.device(f"cuda:{gpu_ids[0]}") if th.cuda.is_available() and gpu_ids else th.device("cpu")

    config, model_config0, class_config = create_config(args.model_name, args.timestep_rp)
    model_config = model_and_diffusion_defaults()
    model_config.update(model_config0)

    return args, config, model_config, class_config


if __name__ == "__main__":
    main()
