import argparse
import os
import shutil

import torch as th

from conditional_sampler import ConditionalSampler
from sample_vubo_helper import SampleVuboHelper
from schedule.schedule_batch import ScheduleBatch
from utils import log_info
from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from config import create_config

def create_args_config():
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        model_name="c64",
        cond_name="cond1",
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # parser.add_argument("--todo", type=str, default='alpha_bar_all, schedule_sample')
    parser.add_argument("--todo", type=str, default='schedule_sample')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7, 6])
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat_times", type=int, default=1, help='run XX times to get avg FID')
    parser.add_argument("--ab_original_dir", type=str,  default='./output7/phase1_ab_original')
    parser.add_argument("--ab_scheduled_dir", type=str, default='./output7/phase2_ab_scheduled')
    parser.add_argument("--ab_summary_dir", type=str,   default='./output7/phase3_ab_summary')
    parser.add_argument("--ss_plan_file", type=str,     default="./output7/vubo_ss_plan.txt")

    parser.add_argument("--timestep_rp", type=int, default=10, help='timestep count')
    parser.add_argument("--method", type=str, default='stsp4')
    parser.add_argument("--num_step_arr", nargs='+', type=int, default=[10, 20, 50])
    parser.add_argument("--method_arr", nargs='+', type=str, default=['ddim', 'stsp4'])

    parser.add_argument("--sample_count", type=int, default=100, help="sample image count")
    parser.add_argument("--sample_batch_size", type=int, default=20, help="0 mean from config file")
    parser.add_argument("--sample_output_dir", type=str, default='output7/generated/')
    parser.add_argument("--fid_input1", type=str, default="./symlink/imagenet64/cls000-050")

    parser.add_argument("--class_lo_hi", nargs='+', type=int, default=[0, 40], help='class id low and high')
    parser.add_argument("--class_img_dir", type=str, default='./symlink/imagenet64')

    # arguments for schedule_batch
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--lp', type=float, default=0.01, help='learning_portion')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aa_low_lambda", type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output7/res_mse_avg_list.txt')
    parser.add_argument("--weight_power", type=float, default=1.0, help='change the weight value')
    parser.add_argument("--beta_schedule", type=str, default="linear")

    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    log_info(f"gpu_ids : {gpu_ids}")
    args.device = th.device(f"cuda:{gpu_ids[0]}") if th.cuda.is_available() and gpu_ids else th.device("cpu")

    config, model_config0, class_config = create_config(args.model_name, args.timestep_rp)
    model_config = model_and_diffusion_defaults()
    model_config.update(model_config0)

    return args, config, model_config, class_config

def aggregate_class_image(args):
    log_info(f"aggregate_class_image()...")
    log_info(f"  fid_input1 : {args.fid_input1}")
    log_info(f"  class_lo_hi: {args.class_lo_hi}")
    if os.path.exists(args.fid_input1) and os.listdir(args.fid_input1):
        return  # trick. we only make aggregation dir if fid_input1 not exist or empty
    c_low, c_high = args.class_lo_hi
    c_img_dir = args.class_img_dir
    log_info(f"  cls_img_dir: {args.class_img_dir}")
    dst_dir = args.fid_input1
    log_info(f"  os.makedirs({dst_dir})")
    os.makedirs(dst_dir)
    f_cnt = 0
    for cid in range(c_low, c_high):
        cid_str = f"cls{cid:03d}"
        src_dir = os.path.join(c_img_dir, cid_str)
        log_info(f"  copying: {src_dir}")
        for f_name in os.listdir(src_dir):
            src_path = os.path.join(src_dir, f_name)
            dst_path = os.path.join(dst_dir, f"{cid_str}_{f_name}")
            shutil.copyfile(src_path, dst_path)
            f_cnt += 1
        # for
    # for
    log_info(f"  copied file: {f_cnt}")
    log_info(f"aggregate_class_image()...done")

def main():
    args, config, model_config, class_config = create_args_config()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"args: {args}")
    a = args.todo
    if a == 'sample':
        log_info(f"{a} ======================================================================")
        cs = ConditionalSampler(args, config, model_config, class_config)
        cs.sample()
    elif a == 'sample_baseline':
        log_info(f"{a} ======================================================================")
        aggregate_class_image(args)
        cs = ConditionalSampler(args, config, model_config, class_config)
        cs.sample_baseline()
    elif a == 'alpha_bar_all':
        log_info(f"{a} ======================================================================")
        cs = ConditionalSampler(args, config, model_config, class_config)
        cs.alpha_bar_all()
    elif a == 'schedule' or a == 'schedule_only':
        log_info(f"{a} ======================================================================")
        sb = ScheduleBatch(args)
        sb.schedule_batch()
    elif a == 'sample_scheduled':
        log_info(f"{a} ======================================================================")
        aggregate_class_image(args)
        runner = ConditionalSampler(args, config, model_config, class_config)
        helper = SampleVuboHelper(args, runner)
        helper.sample_scheduled()
    elif a == 'schedule_sample':
        log_info(f"{a} ======================================================================")
        aggregate_class_image(args)
        runner = ConditionalSampler(args, config, model_config, class_config)
        helper = SampleVuboHelper(args, runner)
        helper.schedule_sample()
    else:
        raise ValueError(f"Unknown todo: {a}")

if __name__ == "__main__":
    main()
