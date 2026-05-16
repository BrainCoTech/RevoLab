"""ProprioAdapt Stage2 student distillation.

Freezes all Stage1 weights except adapt_tconv. Online training: one forward+backward
  per env step. MSE loss between adapt_tconv(proprio_hist).tanh() and frozen
  env_mlp(priv_info).tanh().detach().

Checkpoint: .ckpt extension. Warm-start from Stage1 .pth via strict=False.
  Full resume supports optimizer/agent_steps/best_rewards/rms/sa_ms.

Gotcha — sa_mean_std stays in train() mode (accumulating proprio_hist statistics
  online), while running_mean_std stays in eval() mode (Stage1 stats frozen).
"""
import os
import time
import math
import torch
from termcolor import cprint

from BrainCo_DexHand.algo.hora.utils.misc import AverageScalarMeter, tprint
from BrainCo_DexHand.algo.hora.models.models import ActorCritic
from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter


class ProprioAdapt(object):
    def __init__(self, env, output_dir, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        self.proprio_hist_dim = self.env.prop_hist_len
        self.obs_per_step = self.obs_shape[0] // 3
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'obs_per_step': self.obs_per_step,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, self.obs_per_step)).to(self.device)
        self.sa_mean_std.train()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        writer = SummaryWriter(self.tb_dir)
        self.writer = writer
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.save_frequency = int(self.ppo_config.get('save_frequency', 0))
        # ---- Optim ----
        adapt_params = []
        for name, p in self.model.named_parameters():
            if 'adapt_tconv' in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train(self):
        _t = time.time()
        _last_t = time.time()
        total_iters = max(1, math.ceil(self.max_agent_steps / self.batch_size))
        iter_num = 0

        obs_dict = self.env.reset()
        while self.agent_steps < self.max_agent_steps:
            iter_num += 1
            iter_start_t = time.time()

            learn_start_t = time.time()
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']).detach(),
                'priv_info': obs_dict['priv_info'],
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
            }
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
            loss = ((e - e_gt.detach()) ** 2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            learn_t = time.time() - learn_start_t

            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            collect_start_t = time.time()
            obs_dict, r, done, info = self.env.step(mu)
            for k, v in info.items():
                if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                    self.direct_info[k] = float(v)
            collect_t = time.time() - collect_start_t
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.log_tensorboard()

            if self.save_frequency > 0 and iter_num % self.save_frequency == 0:
                step_m = int(self.agent_steps // 1e6)
                self.save(os.path.join(self.nn_dir, f'{step_m:04d}M'))
                self.save(os.path.join(self.nn_dir, 'model_last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, 'model_best'))
                self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            tprint(info_string)
            print("", flush=True)
            self._print_epoch_log(
                iter_num=iter_num,
                total_iters=total_iters,
                collect_t=collect_t,
                learn_t=learn_t,
                iter_t=time.time() - iter_start_t,
                elapsed=time.time() - _t,
                mean_rewards=mean_rewards,
                mean_lengths=self.mean_eps_length.get_mean(),
            )

    def log_tensorboard(self):
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        is_stage2_ckpt = str(fn).endswith(".ckpt") or ("stage2_nn" in str(fn))
        if is_stage2_ckpt:
            required_keys = ["model", "optimizer", "agent_steps", "best_rewards"]
            missing = [k for k in required_keys if k not in checkpoint]
            if missing:
                raise RuntimeError(
                    f"Stage2 resume failed: missing keys {missing} in checkpoint: {fn}"
                )

            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.agent_steps = int(checkpoint["agent_steps"])
            self.best_rewards = float(checkpoint["best_rewards"])
            if "running_mean_std" in checkpoint:
                self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            if "sa_mean_std" in checkpoint:
                self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
            print(
                f"[INFO] Resumed Stage2: agent_steps={self.agent_steps}, "
                f"best_rewards={self.best_rewards:.4f}",
                flush=True,
            )
            return

        cprint('careful, using non-strict matching', 'red', attrs=['bold'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        if 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'sa_mean_std' in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
        print("[INFO] Warm-start Stage2 from non-resume checkpoint.", flush=True)

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'agent_steps': int(self.agent_steps),
            'best_rewards': float(self.best_rewards),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights['sa_mean_std'] = self.sa_mean_std.state_dict()
        torch.save(weights, f'{name}.ckpt')

    def _print_epoch_log(self, iter_num, total_iters, collect_t, learn_t, iter_t, elapsed, mean_rewards, mean_lengths):
        width = 100
        pad = 30
        fps = int(self.batch_size / max(1e-6, collect_t + learn_t))
        eta_sec = max(0.0, (total_iters - iter_num) * (elapsed / max(1, iter_num)))

        # Collect numeric extras for display
        rew_items = []
        for k in sorted(self.direct_info.keys()):
            v = self.direct_info[k]
            if isinstance(v, (int, float)):
                rew_items.append((k, float(v)))

        header = f" Learning iteration {iter_num}/{total_iters} "
        lines = [
            "#" * width,
            header.center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps} steps/s (collection: {collect_t:.3f}s, learning: {learn_t:.3f}s)",
            f"{'Mean reward:':>{pad}} {mean_rewards:.4f}",
            f"{'Mean episode length:':>{pad}} {mean_lengths:.4f}",
        ]
        for k, v in rew_items:
            lines.append(f"{k + ':':>{pad}} {v:.6f}")
        lines.extend([
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.agent_steps}",
            f"{'Iteration time:':>{pad}} {iter_t:.2f}s",
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}",
        ])
        print("\n".join(lines))
