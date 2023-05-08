import datetime
import os
import shutil
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.utils as tvu
from functools import partial
import torch_fidelity

from guided_diffusion import gaussian_diffusion
from utils import log_info, get_time_ttl_and_eta
from guided_diffusion.script_util import (
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
)

class AlphaBarMapper:
    def __init__(self, noise_schedule, device):
        total_step = 1000
        betas = gaussian_diffusion.get_named_beta_schedule(noise_schedule, total_step)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas, axis=0)  # alpha_bar cumulated product
        self.ab_cump = th.from_numpy(alpha_bars).to(device)
        self.total_step = total_step
        self.noise_schedule = noise_schedule
        timesteps = np.arange(total_step)
        self.timesteps = th.from_numpy(timesteps).to(device)

    @staticmethod
    def interpolate_fn(x, xp, yp):
        """
        A piecewise linear function y = f(x), using xp and yp as key points.
        We implement f(x) in a differentiable way (i.e. applicable for autograd).
        The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp,
         we use the outmost points of xp to define the linear function.)

        Args:
            x: PyTorch tensor with shape [N, C], where N is the batch size,
                C is the number of channels (we use C = 1 for DPM-Solver).
            xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
            yp: PyTorch tensor with shape [C, K].
        Returns:
            The function values f(x), with shape [N, C].
        """
        N, K = x.shape[0], xp.shape[1]
        all_x = th.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
        sorted_all_x, x_indices = th.sort(all_x, dim=2)
        x_idx = th.argmin(x_indices, dim=2)
        cand_start_idx = x_idx - 1
        start_idx = th.where(
            th.eq(x_idx, 0),
            th.tensor(1, device=x.device),
            th.where(th.eq(x_idx, K), th.tensor(K - 2, device=x.device), cand_start_idx),
        )
        end_idx = th.where(th.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
        start_x = th.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
        end_x = th.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
        start_idx2 = th.where(
            th.eq(x_idx, 0),
            th.tensor(0, device=x.device),
            th.where(th.eq(x_idx, K), th.tensor(K - 2, device=x.device), cand_start_idx),
        )
        y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
        start_y = th.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
        end_y = th.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
        cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
        return cand

    def ab2ts(self, alpha_bar):
        """alpha_bar to timestep"""
        if not hasattr(self, '_ab2ts_flag'):
            setattr(self, '_ab2ts_flag', True)
            log_info(f"schedule::ab2ts() called")
        x_arr, y_arr = self.ab_cump, self.timesteps
        x_arr, y_arr = th.flip(x_arr, [0]), th.flip(y_arr, [0])
        x_arr, y_arr = x_arr.reshape((1, -1)), y_arr.reshape((1, -1))
        x = alpha_bar.reshape((-1, 1))
        y = self.interpolate_fn(x, x_arr, y_arr)
        return y

    def ts2ab(self, timestep):
        """timestep to alpha_bar"""
        if not hasattr(self, '_ts2ab_flag'):
            setattr(self, '_ts2ab_flag', True)
            log_info(f"schedule::ts2ab() called")
        x_arr, y_arr = self.timesteps, self.ab_cump
        x_arr, y_arr = x_arr.reshape((1, -1)), y_arr.reshape((1, -1))
        x = timestep.reshape((-1, 1))
        y = self.interpolate_fn(x, x_arr, y_arr)
        return y

class ConditionalSampler:
    """"""
    def __init__(self, args, config, model_config, class_config):
        self.args = args
        self.config = config
        self.model_config = model_config
        self.class_config = class_config
        self.model_name = args.model_name
        self.cond_name  = args.cond_name
        self.timestep_rp = args.timestep_rp
        self.num_step_arr = args.num_step_arr
        self.method = args.method
        self.method_arr = args.method_arr
        self.device = args.device
        self.real_seq = None
        self.noise_schedule = model_config['noise_schedule']
        abm = AlphaBarMapper(self.noise_schedule, self.device)  # alpha_bar mapper
        self.ab_mapper = abm
        log_info(f"ConditionalSampler()...")
        log_info(f"  model_name    : {self.model_name}")
        log_info(f"  cond_name     : {self.cond_name}")
        log_info(f"  timestep_rp   : {self.timestep_rp}")
        log_info(f"  method        : {self.method}")
        log_info(f"  device        : {self.device}")
        log_info(f"  noise_schedule: {self.noise_schedule}")
        log_info(f"  AlphaBarMap() : total_step={abm.total_step}, noise={abm.noise_schedule}")

    @staticmethod
    def load_predefined_aap(f_path: str, meta_dict=None):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        if meta_dict is None:
            meta_dict = {}
        log_info(f"Load file: {f_path}")
        with open(f_path, 'r') as f_ptr:
            lines = f_ptr.readlines()
        cnt_empty = 0
        cnt_comment = 0
        ab_arr = []  # alpha_bar array
        ts_arr = []  # timestep array
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):  # line is like "# order     : 2"
                cnt_comment += 1
                arr = line[1:].strip().split(':')
                key = arr[0].strip()
                if key in meta_dict: meta_dict[key] = arr[1].strip()
                continue
            arr = line.split(':')
            ab, ts = float(arr[0]), float(arr[1])
            ab_arr.append(ab)
            ts_arr.append(ts)
        ab2s = lambda ff: ' '.join([f"{f:8.6f}" for f in ff])
        ts2s = lambda ff: ' '.join([f"{f:10.5f}" for f in ff])
        log_info(f"  cnt_empty  : {cnt_empty}")
        log_info(f"  cnt_comment: {cnt_comment}")
        log_info(f"  cnt_valid  : {len(ab_arr)}")
        log_info(f"  ab[:5]     : [{ab2s(ab_arr[:5])}]")
        log_info(f"  ab[-5:]    : [{ab2s(ab_arr[-5:])}]")
        log_info(f"  ts[:5]     : [{ts2s(ts_arr[:5])}]")
        log_info(f"  ts[-5:]    : [{ts2s(ts_arr[-5:])}]")
        for k, v in meta_dict.items():
            log_info(f"  {k:11s}: {v}")
        return ab_arr, ts_arr

    def create_model_diffusion_classifier(self, ab=None, ts=None):
        args = self.args
        log_info("create_model_diffusion_classifier()...")
        model, diffusion = create_model_and_diffusion(**self.model_config, predefined_ab=ab, predefined_ts=ts)
        use_timesteps = list(diffusion.use_timesteps)
        use_timesteps.sort()
        self.real_seq = use_timesteps
        ab_list = diffusion.alphas_cumprod
        log_info(f"  use_timesteps cnt: {len(use_timesteps)}")
        log_info(f"  alpha_bar     cnt: {len(ab_list)}")
        for i in range(len(ab_list)):
            ts, ab = use_timesteps[i], ab_list[i]
            log_info(f"  ts - a_bar [{i:2d}]  : {ts:5.1f} - {ab:.8f}")

        m_path = os.path.join(args.root_dir, self.config['model_path'])
        log_info(f"  load diff model: {m_path}")
        s_dict = th.load(m_path, map_location='cpu')
        model.load_state_dict(s_dict)
        log_info(f"  load diff model: {m_path} ... done")
        model.requires_grad_(False).eval()
        model.to(self.device)
        log_info(f"  model.to({self.device})")
        if self.model_config['use_fp16']:
            model.convert_to_fp16()
            log_info(f"  model.convert_to_fp16()")
        if len(args.gpu_ids) > 1:
            log_info(f"  torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
            model = th.nn.DataParallel(model, device_ids=args.gpu_ids)

        classifier_config = classifier_defaults()
        classifier_config.update(self.class_config)
        classifier = create_classifier(**classifier_config)
        c_path = os.path.join(args.root_dir, self.config['classifier_path'])
        log_info(f"  load classifier: {m_path}")
        s_dict = th.load(c_path, map_location="cpu")
        classifier.load_state_dict(s_dict)
        log_info(f"  load classifier: {m_path} ... done")
        classifier.to(self.device)
        log_info(f"  classifier.to({self.device})")
        classifier.requires_grad_(False).eval()
        if classifier_config['classifier_use_fp16']:
            classifier.convert_to_fp16()
            log_info(f"  classifier.convert_to_fp16()")
        if len(args.gpu_ids) > 1:
            log_info(f"  torch.nn.DataParallel(classifier, device_ids={args.gpu_ids})")
            classifier = th.nn.DataParallel(classifier, device_ids=args.gpu_ids)
        log_info("create_model_diffusion_classifier()...done")
        return model, diffusion, classifier

    def create_functions(self, model, diffusion, classifier):
        args, config = self.args, self.config
        model_config = self.model_config

        def cond_fn(x, t, y=None, **_):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                grad = th.autograd.grad(selected.sum(), x_in)[0]
                return grad * config['classifier_scale']

        log_info(f"create_functions()...")
        log_info(f"  self.model_name : {self.model_name}")
        log_info(f"  self.cond_name  : {self.cond_name}")
        log_info(f"  self.method     : {self.method}")
        log_info(f"  classifier_scale: {config['classifier_scale']}")
        log_info(f"  class_cond      : {model_config['class_cond']}")

        def model_fn(x, t, y=None):
            assert y is not None
            if gaussian_diffusion._batch_index == 0:
                log_info(f"conditional_sampler::model_fn() t={t[0]:7.3f}")
            return model(x, t, y if model_config['class_cond'] else None)

        if self.method == 'ddim':
            sample_fn = diffusion.ddim_sample_loop
        elif self.method[:4] in ['plms', 'pndm']:
            sample_fn = partial(diffusion.plms_sample_loop, order=int(self.method[4]))
        elif self.method[:4] in ['ltsp', 'ours', 'ltts']:
            sample_fn = partial(diffusion.ltsp_sample_loop, order=int(self.method[4]))
        elif self.method[:4] in ['stsp', 'bchf']:
            sample_fn = partial(diffusion.stsp_sample_loop, order=int(self.method[4]))
        else:
            sample_fn = diffusion.p_sample_loop

        if self.cond_name == 'cond1':
            cond_fn0 = cond_fn
        else:
            cond_fn0 = None
        log_info(f"create_functions()...done")
        return model_fn, cond_fn0, sample_fn

    def save_image(self, sample, classes, b_idx, b_size, b_cnt, time_start):
        sample = ((sample + 1) / 2).clamp(0.0, 1.0)
        img_cnt = len(sample)
        img_path = None
        init_id = b_idx * b_size
        elp, eta = get_time_ttl_and_eta(time_start, b_idx + 1, b_cnt)
        out_dir = self.args.sample_output_dir
        for i in range(img_cnt):
            img_id = init_id + i
            cls_id = classes[i]
            img_path = os.path.join(out_dir, f"{img_id:05d}_c{cls_id:03d}.png")
            tvu.save_image(sample[i], img_path)
        log_info(f"  saved {img_cnt} images: {img_path}. elp:{elp}, eta:{eta}")

    def sample(self, total_num=None, aap_file=None):
        args, config = self.args, self.config
        log_info("sampling()...")
        out_dir = args.sample_output_dir
        if os.path.exists(out_dir):
            # the image file is like 00000_c152.png, where 152 is class ID.
            # And each time the class id may be different. When generating new images, the new
            # image may be 00000_c333.png, and the old one will not be overwritten.
            # Therefore, if we generate 50K images, there may be more images in the folder.
            # This is the reason that we delete folder here.
            log_info(f"shutil.rmtree({out_dir})")
            shutil.rmtree(out_dir)
        log_info(f"os.makedirs({out_dir})")
        os.makedirs(out_dir)
        sample_cnt = total_num or args.sample_count
        batch_size = args.sample_batch_size
        image_size = config['image_size']
        batch_cnt = (sample_cnt - 1) // batch_size + 1
        log_info(f"sample_cnt : {sample_cnt}")
        log_info(f"batch_size : {batch_size}")
        log_info(f"batch_cnt  : {batch_cnt}")
        log_info(f"image_size : {image_size}")
        log_info(f"out_dir    : {out_dir}")
        log_info(f"class_lo_hi: {args.class_lo_hi}")
        log_info(f"timestep_rp: {self.timestep_rp}")
        if aap_file:
            # todo: create diffusion based on aap
            meta_dict = {"num_step": None, "grad_method": None}
            ab_arr, _ = self.load_predefined_aap(aap_file, meta_dict)
            num_step = int(meta_dict['num_step'])
            self.timestep_rp = num_step
            self.model_config['timestep_respacing'] = str(num_step)
            self.method = meta_dict['grad_method']
            ab_tensor = th.tensor(ab_arr, device=self.device)
            ts_tensor = self.ab_mapper.ab2ts(ab_tensor)
            ts_tensor = ts_tensor.squeeze(1)
            model, diffusion, classifier = self.create_model_diffusion_classifier(ab_tensor, ts_tensor)
        else:
            model, diffusion, classifier = self.create_model_diffusion_classifier()
        model_fn, cond_fn, sample_fn = self.create_functions(model, diffusion, classifier)
        cid_low, cid_high = args.class_lo_hi
        time_start = time.time()
        for b_idx in range(batch_cnt):
            gaussian_diffusion._batch_index = b_idx
            log_info(f"B{b_idx:03d}/{batch_cnt}")
            n = batch_size if b_idx + 1 < batch_cnt else sample_cnt - b_idx * batch_size
            classes = th.randint(low=cid_low, high=cid_high, size=(n,), device=self.device)
            sample = sample_fn(
                model_fn,
                (n, 3, image_size, image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs={"y": classes},
                cond_fn=cond_fn,
                impu_fn=None,
                progress=False,  # do not show progress. Then can save output to log files
                device=self.device,
            )
            self.save_image(sample, classes, b_idx, batch_size, batch_cnt, time_start)
        # while
        log_info("sampling complete")

    def config_key_str(self):
        ks = f"s{self.timestep_rp:02d}_{self.method}_{self.model_name}"
        return ks

    def sample_times(self, times=None, aap_file=None):
        args = self.args
        times = times or args.repeat_times
        fid_arr = []
        input1, input2 = args.fid_input1, args.sample_output_dir
        ss = self.config_key_str()
        for i in range(times):
            self.sample(aap_file=aap_file)
            ss = self.config_key_str()  # get ss again as config may change due to aap_file.
            log_info(f"{ss}-{i}/{times} => FID calculating...")
            log_info(f"  input1: {input1}")
            log_info(f"  input2: {input2}")
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=input1,
                input2=input2,
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
                samples_find_deep=True,
            )
            fid = metrics_dict['frechet_inception_distance']
            log_info(f"{ss}-{i}/{times} => FID: {fid:.6f}")
            fid_arr.append(fid)
        # for
        avg, std = np.mean(fid_arr), np.std(fid_arr)
        return ss, avg, std

    def sample_baseline(self):
        def save_result(_msg_arr, _fid_arr):
            with open('./sample_all_result.txt', 'w') as f_ptr:
                [f_ptr.write(f"# {m}\n") for m in _msg_arr]
                [f_ptr.write(f"[{dt}] {a:8.4f} {s:.4f}: {k}\n") for dt, a, s, k in _fid_arr]
            # with
        msg_arr = [f"  num_step_arr   : {self.num_step_arr}",
                   f"  method_arr     : {self.method_arr}"]
        log_info(f"conditional_sampler::sample_baseline()...")
        [log_info(m) for m in msg_arr]
        fid_arr = []
        for timestep_rp in self.num_step_arr:
            self.timestep_rp = timestep_rp
            self.model_config['timestep_respacing'] = str(timestep_rp)
            for method in self.method_arr:
                self.method = method
                key, avg, std = self.sample_times()
                dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                fid_arr.append([dtstr, avg, std, key])
                save_result(msg_arr, fid_arr)
            # for
        # for

    def alpha_bar_all(self):
        def save_ab_file(file_path):
            ts_list = self.real_seq
            ts_list = th.tensor(ts_list).to(self.device)
            ab_list = self.ab_mapper.ts2ab(ts_list)
            ab_list = ab_list.squeeze(1)
            ab_list = ab_list.clamp(1e-6, 1.0)
            if len(ab_list) != self.timestep_rp:
                raise Exception(f"alpha_bar count {len(ab_list)} not match steps {self.timestep_rp}")
            with open(file_path, 'w') as f_ptr:
                f_ptr.write(f"# num_step   : {self.timestep_rp}\n")
                f_ptr.write(f"# grad_method: {self.method}\n")
                f_ptr.write(f"\n")
                f_ptr.write(f"# alpha_bar : index\n")
                for ab, ts in zip(ab_list, ts_list):
                    f_ptr.write(f"{ab:.8f}  : {ts:10.5f}\n")
            # with
        # def
        ab_dir = self.args.ab_original_dir or '.'
        if not os.path.exists(ab_dir):
            log_info(f"os.makedirs({ab_dir})")
            os.makedirs(ab_dir)
        for timestep_rp in self.num_step_arr:
            self.timestep_rp = timestep_rp
            self.model_config['timestep_respacing'] = str(timestep_rp)
            for method in self.method_arr:
                self.method = method
                self.sample(total_num=1)
                key = self.config_key_str()
                f_path = os.path.join(ab_dir, f"{key}.txt")
                save_ab_file(f_path)
                log_info(f"File saved: {f_path}")
            # for
        # for

# class
