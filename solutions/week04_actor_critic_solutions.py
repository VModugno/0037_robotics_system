"""
Instructor solutions for COMP0037 Week 04 (week04_actor_critic).

Reference implementations for the four student TODOs are monkey-patched onto
the lab module, sanity-checked, then actor_critic.main() runs with
--device cpu --episodes 400 (the LAB.md reference configuration) to reproduce
the reference numbers in week04_actor_critic/LAB.md (self-check):

  T1  GaussianActor.sample_action  -> class-attribute assignment
  T2  compute_gae                 -> module-attribute assignment
  T3  cbf_safe_action             -> module-attribute assignment
  T4  A2CAgent.update             -> the TODO is a single line INSIDE the
                                     method, so the ENTIRE method is
                                     re-implemented here (body copied
                                     verbatim from actor_critic.py with the
                                     actor_loss line filled in) and assigned
                                     as a class attribute.

INSTRUCTOR USE ONLY -- students implement the TODOs themselves.
a2c_comparison.png lands in the current working directory. The reference
numbers summary is captured by wrapping actor_critic.train (same signature;
delegates to the original and stores the returned history dict).

torch.manual_seed(0) is set for a reproducible reference run — the scaffold
seeds the env per-episode but never seeds torch, so network init and rsample
would otherwise differ run to run.

Run from the labs/ root:
    pixi run --manifest-path pixi.toml -e week04 python solutions/week04_actor_critic_solutions.py --device cpu --episodes 400
"""
from __future__ import annotations

import os
import sys
import time

LAB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "week04_actor_critic")
)

sys.dont_write_bytecode = True       # no __pycache__ inside the lab folder
sys.path.insert(0, LAB_DIR)

import numpy as np
import torch
import torch.nn as nn

import arm2d_env
import actor_critic as ac
from arm2d_env import DT, JOINT_LIMITS

EPISODES = 400
DEVICE = "cpu"
TORCH_SEED = 0


# ───────────────────────────────────────────────────────── T1 (class attr)
def solved_sample_action(self, obs: np.ndarray) -> np.ndarray:
    """Reference for TODO(T1) — steps exactly as in the scaffold docstring."""
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    dist = self.distribution(x)
    a = dist.rsample()
    return a[0].cpu().numpy()


# ──────────────────────────────────────────────────────── T2 (module attr)
def solved_compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference for TODO(T2) — exactly the docstring recursion:

        delta_t = r_t + gamma * next_values[t] - values[t]
        A_t     = delta_t + gamma * lam * A_{t+1}    (A_T = 0, backwards)
        returns = A + values

    `dones` is unused by the docstring formula: the caller (update) already
    puts 0.0 into next_values on the terminal step, and a success can only
    occur on the LAST step of an episode, so no masking is needed here.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    nv = np.asarray(next_values, dtype=np.float64)
    T = r.shape[0]
    adv = np.zeros(T, dtype=np.float64)
    next_adv = 0.0
    for t in range(T - 1, -1, -1):
        delta = r[t] + gamma * nv[t] - v[t]
        next_adv = delta + gamma * lam * next_adv
        adv[t] = next_adv
    returns = adv + v
    return adv, returns


# ──────────────────────────────────────────────────────── T3 (module attr)
def solved_cbf_safe_action(
    theta: np.ndarray,
    omega: np.ndarray,
    limits: tuple[float, float] = JOINT_LIMITS,
    h_min: float = 0.05,
    dt: float = DT,
) -> tuple[np.ndarray, int]:
    """Reference for TODO(T3) — per-joint projection exactly per docstring:
    margin_i = limit_i - h_min; if |theta_i| + |omega_i|*dt > margin_i AND
    omega_i pushes |theta_i| outward, clip |omega_i| to (margin_i - |theta_i|)/dt
    (floored at 0). Returns (omega_safe, n_clipped).
    """
    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64).copy()
    n_clipped = 0
    for i in range(2):
        margin = limits[i] - h_min
        at = abs(th[i])
        overshoot = at + abs(om[i]) * dt - margin
        outward = th[i] * om[i] > 0.0
        if overshoot > 0.0 and outward:
            om[i] = np.sign(om[i]) * max((margin - at) / dt, 0.0)
            n_clipped += 1
    return om, n_clipped


# ─────────────────────────────────────────────────────── T4 (whole method)
def solved_update(self, tr: dict) -> tuple[float, float]:
    """Reference for TODO(T4) — the ENTIRE A2CAgent.update method, copied
    verbatim from actor_critic.py (lines 203-248) with the single TODO line
    implemented:
        actor_loss = -(logp * adv_t).mean() - self.entropy_coef * entropy
    """
    obs_t = torch.as_tensor(tr["obs"], device=self.device)
    act_t = torch.as_tensor(tr["actions"], device=self.device)
    rew_np = tr["rewards"]
    done_np = tr["dones"]

    values = self.critic(obs_t)                       # (T,), with grad
    with torch.no_grad():
        values_np = values.cpu().numpy()
        if bool(done_np[-1]):
            last_v = 0.0
        else:                                          # truncated: bootstrap
            f = torch.as_tensor(tr["final_obs"], dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            last_v = float(self.critic(f).item())
        next_values_np = np.append(values_np[1:], last_v)

    adv_np, ret_np = ac.compute_gae(rew_np, values_np, next_values_np, done_np,
                                    self.gamma, self.lam)
    adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)   # normalise
    adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
    ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)

    dist = self.distribution_of(obs_t)
    logp = dist.log_prob(act_t).sum(-1)               # (T,)
    entropy = dist.entropy().sum(-1).mean()           # scalar

    # ── TODO(T4) solved: the policy-gradient loss, one line ───────────────
    actor_loss = -(logp * adv_t).mean() - self.entropy_coef * entropy
    # ─────────────────────────────────────────────────────────────────────

    critic_loss = nn.functional.mse_loss(values, ret_t)

    self.actor_opt.zero_grad()
    actor_loss.backward()
    self.actor_opt.step()
    self.critic_opt.zero_grad()
    critic_loss.backward()
    self.critic_opt.step()
    return float(actor_loss.item()), float(critic_loss.item())


# ────────────────────────────────────────────────────────── apply patches
ac.GaussianActor.sample_action = solved_sample_action   # T1
ac.compute_gae = solved_compute_gae                     # T2
ac.cbf_safe_action = solved_cbf_safe_action             # T3
ac.A2CAgent.update = solved_update                      # T4


# ────────────────────────────────────── capture stats by wrapping ac.train
_train_history: dict[str, dict] = {}
_original_train = ac.train


def _capturing_train(env: arm2d_env.Arm2DEnv, agent: ac.A2CAgent,
                     episodes: int = 400, safety: bool = False,
                     eval_every: int = 25) -> dict:
    t0 = time.time()
    history = _original_train(env, agent, episodes=episodes,
                             safety=safety, eval_every=eval_every)
    history["_wall_seconds"] = time.time() - t0
    _train_history["CBF shield (safe)" if safety else "no shield (unsafe)"] = history
    return history


ac.train = _capturing_train


# ─────────────────────────────────────────────────────────── sanity checks
def sanity_checks() -> None:
    # T2 vs an independent nested-loop expansion of GAE
    rng = np.random.default_rng(123)
    T = 7
    r = rng.normal(size=T)
    v = rng.normal(size=T)
    nv = rng.normal(size=T)
    g, l = 0.99, 0.95
    adv, ret = solved_compute_gae(r, v, nv, np.zeros(T), g, l)
    delta = r + g * nv - v
    expected = np.array([
        sum((g * l) ** k * delta[t + k] for k in range(T - t))
        for t in range(T)
    ])
    assert np.allclose(adv, expected), "T2 mismatch vs nested-loop GAE"
    assert np.allclose(ret, adv + v)
    print("T2 compute_gae: matches independent GAE expansion   [OK]")

    # T3 hand-computed cases
    om, n = solved_cbf_safe_action(np.array([2.5, 0.0]), np.array([2.5, 0.5]))
    assert np.allclose(om, [1.0, 0.5]) and n == 1, (om, n)
    om, n = solved_cbf_safe_action(np.array([2.55, 0.0]), np.array([2.5, 0.0]))
    assert np.allclose(om, [0.0, 0.0]) and n == 1, (om, n)   # at margin -> 0
    om, n = solved_cbf_safe_action(np.array([-2.5, 1.0]), np.array([-2.5, 2.5]))
    assert np.allclose(om, [-1.0, 2.5]) and n == 1, (om, n)  # sign preserved
    print("T3 cbf_safe_action: hand-computed cases            [OK]")

    # shield unit conversion (OMEGA_MAX) end-to-end against the env
    env = arm2d_env.Arm2DEnv()
    env.reset(seed=0)
    env.theta = np.array([2.55, 0.0])                    # exactly at the margin
    a_sh, n = ac.safety_shield(env.thetas, np.array([1.0, 0.0]))
    _, _, _, info = env.step(a_sh)
    assert not info["limit_hit"] and n == 1, (a_sh, info)
    env.reset(seed=0)
    env.theta = np.array([2.55, 0.0])
    _, _, _, info = env.step(np.array([1.0, 0.0]))
    assert info["limit_hit"], "unshielded full action must clamp"
    print("safety_shield OMEGA_MAX unit conversion            [OK]")


def summary() -> None:
    print("\n" + "=" * 88)
    print(f"REFERENCE NUMBERS   (episodes={EPISODES}, device={DEVICE}, "
          f"torch.manual_seed({TORCH_SEED}))")
    print("=" * 88)
    print(f"{'config':<20}{'return(last20)':>16}{'cum hits':>10}"
          f"{'hits/ep(last20)':>17}{'greedy success':>16}{'wall(s)':>9}")
    for label, h in _train_history.items():
        ret20 = float(np.mean(h["returns"][-20:]))
        cum = int(np.sum(h["hits"]))
        hits20 = float(np.mean(h["hits"][-20:]))
        sr = h["success"][-1][1] if h["success"] else float("nan")
        print(f"{label:<20}{ret20:>16.2f}{cum:>10d}{hits20:>17.2f}"
              f"{sr:>16.0%}{h['_wall_seconds']:>9.0f}")
    print("=" * 88)
    png = os.path.abspath("a2c_comparison.png")
    print(f"plot: {png}  (exists={os.path.exists(png)})")


def main() -> None:
    print(f"python {sys.version.split()[0]}   torch {torch.__version__}   "
          f"numpy {np.__version__}")
    sanity_checks()
    torch.manual_seed(TORCH_SEED)   # scaffold itself never seeds torch
    sys.argv = ["actor_critic.py", "--device", DEVICE, "--episodes", str(EPISODES)]
    try:
        ac.main()
    except Exception as exc:   # keep captured stats even if plotting fails
        print(f"\n[!] actor_critic.main() raised {type(exc).__name__}: {exc}")
        print("[!] reporting stats captured during train() anyway")
    summary()


if __name__ == "__main__":
    main()
