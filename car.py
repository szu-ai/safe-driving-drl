#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gym
    from gym import spaces
except Exception:
    import gymnasium as gym
    from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# CARLA import (0.9.15 robust)
# ======================================================
def _candidate_carla_roots() -> List[str]:
    env_root = os.environ.get("CARLA_ROOT", "").strip()
    candidates = [
        env_root,
        os.path.expanduser("~/carla"),
        os.path.expanduser("~/CARLA_0.9.15"),
        os.path.expanduser("~/CARLA_0.9.15/LinuxNoEditor"),
        os.path.expanduser("~/CARLA_0.9.14"),
        os.path.expanduser("~/CARLA_0.9.14/LinuxNoEditor"),
        os.path.expanduser("~/CARLA_0.9.13"),
        os.path.expanduser("~/CARLA_0.9.13/LinuxNoEditor"),
    ]
    return [p for p in candidates if p and os.path.exists(p)]

def _setup_carla_pythonapi() -> None:
    """
    Import CARLA without letting helper-source folders shadow a working install.

    Order:
    1) If `import carla` already works, keep it.
    2) Otherwise, try CARLA wheel/egg from PythonAPI/carla/dist.
    3) Do NOT add helper folders here; add them later after carla imports.
    """
    try:
        import carla  # noqa: F401
        return
    except Exception:
        pass

    pymaj, pymin = sys.version_info.major, sys.version_info.minor
    searched_roots: List[str] = []
    searched_dist_dirs: List[str] = []
    tried_pkgs: List[str] = []

    for root in _candidate_carla_roots():
        searched_roots.append(root)

        pythonapi = os.path.join(root, "PythonAPI")
        carla_pkg_dir = os.path.join(pythonapi, "carla")
        dist_dir = os.path.join(carla_pkg_dir, "dist")

        if not os.path.isdir(dist_dir):
            continue

        searched_dist_dirs.append(dist_dir)

        pkg_patterns = [
            os.path.join(dist_dir, f"carla-*-cp{pymaj}{pymin}-*.whl"),
            os.path.join(dist_dir, f"carla-*-py{pymaj}.{pymin}-linux-x86_64.egg"),
            os.path.join(dist_dir, "carla-*.whl"),
            os.path.join(dist_dir, "carla-*.egg"),
        ]

        matches: List[str] = []
        for pat in pkg_patterns:
            matches.extend(glob.glob(pat))

        for pkg in sorted(set(matches), reverse=True):
            tried_pkgs.append(pkg)
            if pkg not in sys.path:
                sys.path.insert(0, pkg)
            try:
                import carla  # noqa: F401
                return
            except Exception:
                try:
                    sys.path.remove(pkg)
                except ValueError:
                    pass

    raise ImportError(
        "Could not import CARLA PythonAPI.\n"
        f"Python version: {pymaj}.{pymin}\n"
        f"Searched CARLA roots: {searched_roots}\n"
        f"Searched dist dirs: {searched_dist_dirs}\n"
        f"Tried packages: {tried_pkgs}\n\n"
        "Your previous run indicates CARLA was importable before this patch, so the usual cause here is\n"
        "that helper-source paths were added too early and shadowed the working CARLA package.\n\n"
        "Fix options:\n"
        "1) Keep this function exactly as shown\n"
        "2) Make sure your working environment can do: python3 -c 'import carla; print(carla)'\n"
        "3) If needed, export CARLA_ROOT to the real CARLA install root"
    )

def _setup_carla_helper_paths() -> None:
    """
    Add helper-module paths only AFTER `import carla` succeeds.

    Needed for:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    """
    for root in _candidate_carla_roots():
        pythonapi = os.path.join(root, "PythonAPI")
        carla_pkg_dir = os.path.join(pythonapi, "carla")

        # `agents` lives under PythonAPI/carla/agents
        if os.path.isdir(carla_pkg_dir) and carla_pkg_dir not in sys.path:
            sys.path.insert(0, carla_pkg_dir)

        # Optional helper path; safe after `carla` is already imported
        if os.path.isdir(pythonapi) and pythonapi not in sys.path:
            sys.path.append(pythonapi)

_setup_carla_pythonapi()

import carla  # noqa: E402

_setup_carla_helper_paths()

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception as e1:
    try:
        from carla.agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
    except Exception as e2:
        print(f"[WARN] GlobalRoutePlanner import failed: {e1}")
        print(f"[WARN] Fallback GlobalRoutePlanner import failed: {e2}")
        GlobalRoutePlanner = None
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


# ======================================================
# Config
# ======================================================
@dataclass
class Config:
    fps: int = 20
    client_timeout: float = 120.0
    tm_port: int = 8000
    seed: int = 42
    debug_mode: bool = True
    debug_step_freq: int = 20
    follow_ego_view: bool = True
    no_rendering_mode: bool = False

    car_name: str = "vehicle.tesla.model3"
    fixed_color: str = "255,0,0"

    fixed_weather: str = "night_rain_fog"
    target_weather: str = "mixed"
    use_safety_shield: bool = False
    mode_name: str = "eval"

    npc_min: int = 0
    npc_max: int = 2
    max_entity_obs: int = 10
    entity_max_dist: float = 60.0

    use_fixed_destination: bool = True
    fixed_goal_index: int = 25
    target_goal_index: int = -1
    # Paper Sec 3.2: route R of ~200 m along lane centerlines.
    min_route_length_m: float = 185.0          # hard floor; reject shorter routes
    candidate_goal_min_dist_m: float = 185.0   # goal must be ≥185 m from spawn
    candidate_goal_max_tries: int = 80
    route_success_pct: float = 95.0            # 95% of 200 m = 190 m traversed
    strict_goal_route: bool = True
    allow_fallback_route: bool = False
    route_target_length_m: float = 200.0       # target arc-length (paper)
    route_target_tolerance_m: float = 12.0     # strict ±12 m window
    route_soft_min_length_m: float = 188.0     # soft lower bound
    route_soft_max_length_m: float = 215.0     # soft upper bound
    prefer_auto_length_route: bool = True
    enforce_route_length_for_fixed_goal: bool = True   # paper: fixed 200 m route
    max_reset_start_progress_pct: float = 20.0
    max_reset_start_dL_m: float = 3.5

    sem_dim: int = 3
    eps_min: float = 1.5
    eps_max: float = 4.0

    tau_d: float = 1.0
    tau_p: float = 12.0
    # Paper Eq. 11: rp = tanh(Δs/τs). Smaller τs = more signal at small steps.
    # At 18 km/h, 20 Hz: Δs ≈ 0.25 m → tanh(0.25/1.5) ≈ 0.16 vs 0.12 at τs=2.
    tau_s: float = 1.5
    tau_v: float = 5.0   # paper Eq. 11 velocity-projection scale (v̂_t / τv)
    k_l: float = 0.8
    k_p: float = 0.6
    k_r: float = 0.7
    k_j: float = 0.05
    k_delta: float = 0.10

    # Paper Eq. 9: r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u, Σw = 1
    w_s: float = 0.45
    w_p: float = 0.30
    w_c: float = 0.15
    w_u: float = 0.10

    tl_near_dist: float = 22.0
    tl_stop_dist: float = 10.0
    min_ttc_s: float = 1.8
    terminate_on_collision: bool = True
    terminate_on_offroad: bool = True

    n_critics: int = 5
    calib_temperature: float = 1.0
    action_dim: int = 3
    gamma: float = 0.99
    target_tau: float = 5e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    beta0: float = 1.0
    lambda_ent: float = 1.0
    lambda_transfer: float = 1.0
    lambda_alpha_align: float = 1.0
    lambda_u_align: float = 1.0
    replay_size: int = 200000
    batch_size: int = 512
    maml_inner_lr: float = 1e-4
    maml_inner_steps: int = 1
    maml_meta_step_size: float = 1.0
    eval_every_steps: int = 10000
    calibrate_every_steps: int = 10000
    calib_num_batches: int = 8
    calib_quantile_lo: float = 0.05
    calib_quantile_hi: float = 0.95
    auto_beta0_selection: bool = True
    beta0_candidates: Tuple[float, ...] = (0.5, 1.0)
    train_npc_min: int = 8
    train_npc_max: int = 20
    adapt_episodes: int = 100

    obs_miss_base: float = 0.08
    obs_pos_noise_m: float = 0.35
    obs_vel_noise_ms: float = 0.45
    tl_obs_noise_m: float = 1.25

    route_point_spacing_m: float = 2.0
    lookahead_base_m: float = 6.0
    lookahead_speed_gain: float = 0.35
    route_steer_kp: float = 2.6
    route_cte_kp: float = 1.95
    route_speed_kp: float = 0.13
    target_speed_kmh: float = 18.0
    min_target_speed_kmh: float = 5.5
    curve_speed_penalty_kmh: float = 11.0
    max_safe_steer_at_speed: float = 0.24
    hard_speed_cap_kmh: float = 21.0
    route_override_dL_m: float = 0.95
    route_hard_dL_m: float = 1.80
    route_override_heading_rad: float = 0.22
    route_hard_heading_rad: float = 0.38

    min_throttle_deadzone: float = 0.05
    min_brake_deadzone: float = 0.05
    low_speed_steer_limit: float = 0.18
    max_throttle_step: float = 0.06
    max_brake_step: float = 0.08
    max_steer_step: float = 0.08

    stuck_speed_kmh: float = 1.0
    stuck_steps_threshold: int = 18
    release_duration_steps: int = 25
    release_throttle: float = 0.42
    stuck_terminate_steps: int = 250

    front_vehicle_max_dist: float = 25.0
    front_vehicle_block_dist: float = 7.5
    front_vehicle_soft_block_dist: float = 12.0
    front_vehicle_soft_block_speed_kmh: float = 1.0

    goal_reach_dist_m: float = 15.0           # success if Euclidean dist to goal ≤ 15 m
    goal_reach_remaining_s_m: float = 20.0    # and arc-length remaining ≤ 20 m

    max_route_deviation_m: float = 5.5
    offroute_grace_steps: int = 22
    offroute_heading_gate_rad: float = 0.35
    strong_offroute_factor: float = 2.0

    route_search_back: int = 3
    route_search_ahead: int = 40

    # Paper-faithful dense reward uses no terminal bonus/penalty terms.
    # These are kept as zeroed compatibility fields for older checkpoints/configs.
    goal_bonus: float = 0.0
    collision_penalty: float = 0.0
    offroute_penalty: float = 0.0
    timeout_penalty: float = 0.0

    blocked_timeout_extension_steps: int = 600
    blocked_timeout_speed_kmh: float = 1.0
    near_goal_timeout_extension_steps: int = 1000   # extra steps when within near_goal_remaining_s_m
    near_goal_remaining_s_m: float = 30.0            # "near goal" zone starts at 30 m remaining
    coast_speed_band_kmh: float = 1.5
    soft_brake_speed_excess_kmh: float = 4.0
    brake_release_threshold: float = 0.12
    center_deadband_m: float = 0.06
    center_push_gain: float = 0.55
    center_push_hard_gain: float = 0.85
    center_push_start_m: float = 0.45
    route_soft_dL_m: float = 0.85
    route_hard_dL_m_tight: float = 1.45
    # Action blending: paper uses pure RL policy; in practice a small route-guidance
    # blend is applied only when the vehicle drifts far from the lane centre.
    # Base blend of 0.20 means 80 % policy / 20 % guidance in nominal conditions.
    steer_guidance_blend: float = 0.68
    policy_route_blend_base: float = 0.20   # nominal: 80% policy (paper-faithful)
    policy_route_blend_bad: float = 0.48    # large CTE / heading error
    policy_route_blend_hard: float = 0.70   # extreme deviation / wrong lane
    open_road_brake_soft_limit: float = 0.06
    open_road_brake_curve_limit: float = 0.14
    caution_ttc_s: float = 2.8
    hard_ttc_s: float = 1.8

    spectator_distance_m: float = 8.0
    spectator_height_m: float = 4.0
    spectator_pitch_deg: float = -15.0

    enable_collision_sensor: bool = True
    enable_lane_invasion_sensor: bool = False
    cleanup_sleep_s: float = 0.10
    tick_retry_count: int = 3
    tick_retry_sleep_s: float = 0.50
    post_spawn_settle_ticks: int = 2
    warmup_reset_ticks: int = 3
    rebuild_env_on_reset_failure: bool = True
    destroy_stale_owned_actors_on_reset: bool = True
    destroy_stale_spawn_blockers: bool = True
    spawn_blocker_radius_m: float = 2.5
    spawn_retry_lift_m: float = 0.35

    skip_cross_town_periodic_eval: bool = True
    max_episode_steps: int = 4200
    out_dir: str = "./culrt_carla_0915_aligned"

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    @property
    def edge_dim(self) -> int:
        return 2 + 2 + self.sem_dim + 2 + 1

    @property
    def max_entities(self) -> int:
        return self.max_entity_obs

    @property
    def scalar_dim(self) -> int:
        return 13

    @property
    def model_dir(self) -> str:
        return os.path.join(self.out_dir, "models")

    @property
    def result_dir(self) -> str:
        return os.path.join(self.out_dir, "results")


CFG = Config()


# ======================================================
# Utilities
# ======================================================
def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vec3_length(v: carla.Vector3D) -> float:
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def distance2d(a: carla.Location, b: carla.Location) -> float:
    dx = float(a.x - b.x)
    dy = float(a.y - b.y)
    return float(math.sqrt(dx * dx + dy * dy))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def point_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, float]:
    ab = b - a
    ab2 = float(np.dot(ab, ab) + 1e-9)
    t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
    proj = a + t * ab
    dist = float(np.linalg.norm(p - proj))
    seg_len = float(np.linalg.norm(ab))
    return proj, dist, seg_len * t


def apply_weather(world: carla.World, mode: str) -> None:
    if mode == "night_rain_fog":
        weather = carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=90.0,
            precipitation_deposits=85.0,
            wind_intensity=30.0,
            sun_altitude_angle=-25.0,
            fog_density=40.0,
            fog_distance=8.0,
            wetness=100.0,
        )
    elif mode == "mixed":
        weather = random.choice(
            [
                carla.WeatherParameters(
                    cloudiness=80.0,
                    precipitation=60.0,
                    precipitation_deposits=55.0,
                    wind_intensity=20.0,
                    sun_altitude_angle=-20.0,
                    fog_density=20.0,
                    fog_distance=20.0,
                    wetness=70.0,
                ),
                carla.WeatherParameters(
                    cloudiness=50.0,
                    precipitation=20.0,
                    precipitation_deposits=20.0,
                    wind_intensity=10.0,
                    sun_altitude_angle=10.0,
                    fog_density=5.0,
                    fog_distance=60.0,
                    wetness=20.0,
                ),
                carla.WeatherParameters(
                    cloudiness=95.0,
                    precipitation=80.0,
                    precipitation_deposits=75.0,
                    wind_intensity=35.0,
                    sun_altitude_angle=-10.0,
                    fog_density=25.0,
                    fog_distance=15.0,
                    wetness=90.0,
                ),
            ]
        )
    else:
        weather = carla.WeatherParameters.Default
    world.set_weather(weather)


def get_fog_norm(world: carla.World) -> float:
    try:
        return float(np.clip(float(world.get_weather().fog_density) / 100.0, 0.0, 1.0))
    except Exception:
        return 0.0


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_module_state_compat(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load weights while tolerating scalar-state expansion."""
    model_state = module.state_dict()
    patched: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            patched[key] = value
            continue
        if key.endswith('mlp_scal.0.weight') and value.ndim == 2 and target.ndim == 2 and value.shape[0] == target.shape[0]:
            new_weight = target.clone()
            cols = min(value.shape[1], target.shape[1])
            new_weight[:, :cols] = value[:, :cols]
            patched[key] = new_weight
    module.load_state_dict(patched, strict=False)


def resolve_existing_path(path_text: str) -> str:
    path_text = str(path_text).strip()
    if not path_text:
        return path_text
    if os.path.isabs(path_text):
        return path_text

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(path_text),
        os.path.abspath(os.path.join(script_dir, path_text)),
        os.path.abspath(os.path.join(os.path.dirname(script_dir), path_text)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def resolve_output_dir(path_text: str) -> str:
    path_text = str(path_text).strip()
    if not path_text:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(script_dir, "culrt_carla_0915_aligned"))

    if os.path.isabs(path_text):
        return path_text

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prefer output relative to the script location.
    script_relative = os.path.abspath(os.path.join(script_dir, path_text))
    cwd_relative = os.path.abspath(path_text)

    if os.path.exists(script_relative):
        return script_relative
    if os.path.exists(cwd_relative):
        return cwd_relative

    return script_relative



def make_server_error_info(action: np.ndarray, reason: str = "server_error") -> Dict[str, object]:
    return {
        "collision": False,
        "off_road": False,
        "off_route": False,
        "timeout": True,
        "success": False,
        "goal_reached": False,
        "term_reason": reason,
        "route_completion_pct": 0.0,
        "goal_dist": 0.0,
        "goal_euclid": 0.0,
        "distance_driven_m": 0.0,
        "intervention_rate": 0.0,
        "dL": 0.0,
        "lane_head": 0.0,
        "time_to_conflict": 999.0,
        "applied_action": np.asarray(action, dtype=np.float32),
        "rs": 0.0,
        "rp": 0.0,
        "rc": 0.0,
        "muA": 0.5,
    }


# ======================================================
# Environment
# ======================================================
class CarlaReliableTransferEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2200,
        town_name: str = "Town02",
        fixed_spawn_index: int = 0,
        fixed_goal_index: Optional[int] = None,
        weather_mode: str = "night_rain_fog",
        cfg: Config = CFG,
    ):
        super().__init__()
        self.cfg = cfg
        self.host = host
        self.port = port
        self.requested_town_name = town_name
        self.fixed_spawn_index = fixed_spawn_index
        self.fixed_goal_index = fixed_goal_index if fixed_goal_index is not None else cfg.fixed_goal_index
        self.weather_mode = weather_mode

        self.client = carla.Client(host, port)
        self.client.set_timeout(cfg.client_timeout)
        self.world: carla.World = self._load_world_with_fallback(town_name)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.original_settings = self.world.get_settings()
        self._apply_sync_settings()

        self.tm = self.client.get_trafficmanager(cfg.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(cfg.seed)

        self.spawn_points = self.map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in the loaded CARLA map.")

        self.vehicle: Optional[carla.Vehicle] = None
        self.npcs: List[carla.Vehicle] = []
        self.sensor_list: List[carla.Actor] = []
        self.collision_events: List[object] = []
        self.lane_invasion_events: List[object] = []
        
        self._teardown_in_progress = False
        self._episode_live = False
        self._terminal_reason = ""

        self.episode_steps = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_steer = 0.0
        self.prev_acc = np.zeros(2, dtype=np.float32)
        self.prev_loc: Optional[carla.Location] = None
        self.distance_driven_m = 0.0

        self.stuck_steps = 0
        self.release_steps_left = 0
        self.offroute_steps = 0
        self.prev_route_s: Optional[float] = None
        self.safety_interventions = 0
        self.blocked_steps_credit = 0

        self.route_xy: List[Tuple[float, float]] = []
        self.route_wps: List[carla.Waypoint] = []
        self.route_cumdist: List[float] = []
        self.goal_transform: Optional[carla.Transform] = None
        self.goal_spawn_transform: Optional[carla.Transform] = None
        self.current_goal_index: Optional[int] = None
        self.current_spawn_index: Optional[int] = None
        self.grp = None
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "scalars": spaces.Box(low=-10.0, high=10.0, shape=(cfg.scalar_dim,), dtype=np.float32),
                "edges": spaces.Box(low=-10.0, high=10.0, shape=(cfg.max_entities, cfg.edge_dim), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(cfg.max_entities,), dtype=np.float32),
            }
        )

        self._safe_tick(label="env_init")
        if self.cfg.debug_mode:
            print(
                f"[OK] Connected to CARLA | host={self.host} port={self.port} "
                f"town={self.town_name} sync=True dt={self.cfg.dt:.3f}s weather={self.weather_mode} "
                f"fixed_goal_index={self.fixed_goal_index}"
            )

    def _available_map_names(self) -> List[str]:
        try:
            return list(self.client.get_available_maps())
        except BaseException:
            return []

    @staticmethod
    def _basename_map(map_path: str) -> str:
        if not map_path:
            return map_path
        return map_path.split("/")[-1].split(".")[0]

    def _resolve_town_name(self, requested: str) -> str:
        available = self._available_map_names()
        available_basenames = {self._basename_map(x): x for x in available}
        preferred: List[str] = []
        if requested:
            preferred.append(requested)
            if requested.endswith("_Opt"):
                preferred.append(requested.replace("_Opt", ""))
            else:
                preferred.append(f"{requested}_Opt")
        preferred.extend(["Town02", "Town03", "Town10HD_Opt", "Town10HD"])
        for name in preferred:
            if name in available_basenames:
                return available_basenames[name]
        return available[0] if available else requested

    def _load_world_with_fallback(self, requested_town: str) -> carla.World:
        resolved = self._resolve_town_name(requested_town)
        try:
            world = self.client.load_world(resolved, reset_settings=False)
            self.town_name = self._basename_map(resolved)
            return world
        except Exception as e:
            print(f"[WARN] Failed to load {resolved}: {e}")
            fallback = self._resolve_town_name("Town02")
            world = self.client.load_world(fallback, reset_settings=False)
            self.town_name = self._basename_map(fallback)
            return world

    def _apply_sync_settings(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.cfg.dt
        settings.no_rendering_mode = bool(self.cfg.no_rendering_mode)
        self.world.apply_settings(settings)

    def _restore_world_settings(self) -> None:
        try:
            self.tm.set_synchronous_mode(False)
        except BaseException:
            pass
        try:
            restored = self.world.get_settings()
            restored.synchronous_mode = False
            restored.fixed_delta_seconds = None
            restored.no_rendering_mode = self.original_settings.no_rendering_mode
            self.world.apply_settings(restored)
        except BaseException:
            pass

    def _make_spectator_transform(self, vehicle_tf: carla.Transform) -> carla.Transform:
        yaw = math.radians(vehicle_tf.rotation.yaw)
        cam_loc = carla.Location(
            x=vehicle_tf.location.x - self.cfg.spectator_distance_m * math.cos(yaw),
            y=vehicle_tf.location.y - self.cfg.spectator_distance_m * math.sin(yaw),
            z=vehicle_tf.location.z + self.cfg.spectator_height_m,
        )
        cam_rot = carla.Rotation(
            pitch=self.cfg.spectator_pitch_deg,
            yaw=vehicle_tf.rotation.yaw,
            roll=0.0,
        )
        return carla.Transform(cam_loc, cam_rot)

    def _snap_spectator_to_ego(self) -> None:
        if self.vehicle is None:
            return
        try:
            spectator = self.world.get_spectator()
            spectator.set_transform(self._make_spectator_transform(self.vehicle.get_transform()))
        except BaseException:
            pass

    def _update_spectator(self) -> None:
        if self.vehicle is None or not self.cfg.follow_ego_view:
            return
        self._snap_spectator_to_ego()

    def _safe_tick(self, label: str = "tick", raise_on_fail: bool = True) -> bool:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.tick_retry_count + 1):
            try:
                self.world.tick()
                return True
            except BaseException as e:
                last_err = Exception(str(e))
                if self.cfg.debug_mode:
                    print(f"[WARN] {label} failed attempt {attempt}/{self.cfg.tick_retry_count}: {e}")
                time.sleep(self.cfg.tick_retry_sleep_s)
        if raise_on_fail and last_err is not None:
            raise RuntimeError(f"CARLA world tick failed during {label}: {last_err}")
        return False

    def _destroy_actor(self, actor: carla.Actor) -> None:
        if actor is None:
            return
        try:
            if getattr(actor, "is_alive", False):
                actor.destroy()
        except BaseException:
            pass

    def _stop_sensor_only(self, actor: carla.Actor) -> None:
        if actor is None:
            return
        try:
            if getattr(actor, "is_alive", False):
                actor.stop()
        except BaseException:
            pass

    def _batch_destroy_actors(self, actors: List[carla.Actor]) -> None:
        """
        Safe destroy for synchronous mode:
        - detach TM first for vehicles
        - destroy one-by-one
        - tick between small groups
        """
        alive: List[carla.Actor] = [a for a in actors if self._actor_is_alive(a)]
        if not alive:
            return

        for actor in alive:
            try:
                if "vehicle." in getattr(actor, "type_id", ""):
                    self._safe_set_autopilot(actor, False)
            except BaseException:
                pass

        self._drain_world_ticks(1, "batch_detach_tm", sleep_s=0.02)

        group = 2
        for start in range(0, len(alive), group):
            chunk = alive[start:start + group]
            for actor in chunk:
                self._safe_destroy_one(actor, sleep_s=0.03)
            self._drain_world_ticks(1, f"batch_destroy_{start}", sleep_s=0.02)


    def _role_name_of_actor(self, actor: carla.Actor) -> str:
        try:
            return str(actor.attributes.get("role_name", "")).strip().lower()
        except BaseException:
            return ""

    def _spawn_transform_variants(self, tf: carla.Transform) -> List[carla.Transform]:
        base = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z),
            carla.Rotation(tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
        )
        dz = float(max(self.cfg.spawn_retry_lift_m, 0.05))
        variants = [base]
        for mul in (1.0, 2.0):
            variants.append(
                carla.Transform(
                    carla.Location(base.location.x, base.location.y, base.location.z + dz * mul),
                    carla.Rotation(base.rotation.pitch, base.rotation.yaw, base.rotation.roll),
                )
            )
        return variants

    def _destroy_residual_owned_actors(self) -> int:
        actors_to_destroy: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, 'filter'):
                iterable = list(actors.filter('vehicle.*'))
            else:
                iterable = [a for a in actors if 'vehicle.' in getattr(a, 'type_id', '')]
            for actor in iterable:
                if actor is None:
                    continue
                try:
                    if self.vehicle is not None and actor.id == self.vehicle.id:
                        continue
                except BaseException:
                    pass
                role = self._role_name_of_actor(actor)
                if role in {'hero', 'autopilot'}:
                    actors_to_destroy.append(actor)
        except BaseException:
            return 0
        if not actors_to_destroy:
            return 0
        self._batch_destroy_actors(actors_to_destroy)
        time.sleep(self.cfg.cleanup_sleep_s)
        self._safe_tick(label='destroy_residual_owned_actors', raise_on_fail=False)
        return len(actors_to_destroy)

    def _clear_spawn_blockers(self, tf: carla.Transform, radius: Optional[float] = None) -> int:
        radius = float(self.cfg.spawn_blocker_radius_m if radius is None else radius)
        actors_to_destroy: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, 'filter'):
                iterable = list(actors.filter('vehicle.*'))
            else:
                iterable = [a for a in actors if 'vehicle.' in getattr(a, 'type_id', '')]
            for actor in iterable:
                if actor is None:
                    continue
                try:
                    if self.vehicle is not None and actor.id == self.vehicle.id:
                        continue
                except BaseException:
                    pass
                try:
                    if distance2d(actor.get_location(), tf.location) <= radius:
                        actors_to_destroy.append(actor)
                except BaseException:
                    continue
        except BaseException:
            return 0
        if not actors_to_destroy:
            return 0
        self._batch_destroy_actors(actors_to_destroy)
        time.sleep(self.cfg.cleanup_sleep_s)
        self._safe_tick(label='clear_spawn_blockers', raise_on_fail=False)
        return len(actors_to_destroy)

    def _cleanup_episode_actors(self) -> None:
        """
        Robust synchronous cleanup.

        Order:
        1. Mark teardown active, stop sensors, detach TM.
        2. Drain queued callbacks.
        3. Destroy sensors.
        4. Destroy ego.
        5. Destroy NPCs after TM detach.
        6. Final settle ticks.
        """
        self._begin_episode_teardown(reason="cleanup")

        sensors = list(self.sensor_list)
        self.sensor_list.clear()

        ego = self.vehicle
        self.vehicle = None

        npcs = list(self.npcs)
        self.npcs.clear()

        # Extra drain after stop() because goal often arrives at nonzero speed.
        self._drain_world_ticks(3, "cleanup_drain_pre", sleep_s=0.03)

        # Destroy sensors first.
        for sensor in sensors:
            self._stop_sensor_only(sensor)
        self._drain_world_ticks(1, "cleanup_post_sensor_stop", sleep_s=0.02)

        for sensor in sensors:
            self._safe_destroy_one(sensor, sleep_s=0.03)

        self.collision_events.clear()
        self.lane_invasion_events.clear()
        self._drain_world_ticks(2, "cleanup_post_sensor_destroy", sleep_s=0.03)

        # Destroy ego next.
        if self._actor_is_alive(ego):
            self._safe_set_autopilot(ego, False)
            try:
                ego.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False,
                    )
                )
            except BaseException:
                pass

        self._safe_destroy_one(ego, sleep_s=0.05)
        self._drain_world_ticks(2, "cleanup_post_ego", sleep_s=0.03)

        # Detach TM from NPCs before destroying them.
        alive_npcs = [npc for npc in npcs if self._actor_is_alive(npc)]
        for npc in alive_npcs:
            self._safe_set_autopilot(npc, False)

        self._drain_world_ticks(2, "cleanup_post_tm_detach", sleep_s=0.03)

        # Destroy NPCs in small chunks.
        group = 2
        for start in range(0, len(alive_npcs), group):
            chunk = alive_npcs[start:start + group]
            for npc in chunk:
                self._safe_destroy_one(npc, sleep_s=0.03)
            self._drain_world_ticks(1, f"cleanup_post_npc_{start}", sleep_s=0.03)

        self._drain_world_ticks(3, "cleanup_final", sleep_s=0.03)

        try:
            time.sleep(max(self.cfg.cleanup_sleep_s, 0.05))
        except BaseException:
            pass

        self._terminal_reason = ""
        self._teardown_in_progress = False

    def _get_global_planner(self):
        if GlobalRoutePlanner is None:
            return None
        if self.grp is None:
            self.grp = GlobalRoutePlanner(self.map, self.cfg.route_point_spacing_m)
        return self.grp
        
    def _rebuild_local_route_from_current_pose(self) -> bool:
        if self.vehicle is None:
            return False

        cur_loc = self.vehicle.get_location()
        cur_wp = self.map.get_waypoint(
            cur_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if cur_wp is None:
            return False

        self.route_xy = []
        self.route_wps = []
        self.route_cumdist = []
        self.current_goal_index = None
        self.goal_spawn_transform = None
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        last_wp = self._build_route_polyline_local(cur_wp, ds=self.cfg.route_point_spacing_m)
        self._compute_route_cumdist()

        goal_loc = last_wp.transform.location
        goal_rot = last_wp.transform.rotation
        self.goal_transform = carla.Transform(
            carla.Location(goal_loc.x, goal_loc.y, goal_loc.z + 0.5),
            goal_rot,
        )
        return len(self.route_wps) >= 2

    def _route_length_from_wps(self, wps: Sequence[carla.Waypoint]) -> float:
        if len(wps) < 2:
            return 0.0
        s = 0.0
        for i in range(len(wps) - 1):
            s += distance2d(wps[i].transform.location, wps[i + 1].transform.location)
        return float(s)

    def _route_length_ok(self, route_len: float, strict: bool = False) -> bool:
        route_len = float(route_len)
        if strict:
            return abs(route_len - self.cfg.route_target_length_m) <= self.cfg.route_target_tolerance_m
        return self.cfg.route_soft_min_length_m <= route_len <= self.cfg.route_soft_max_length_m

    def _route_length_score(self, route_len: float) -> float:
        return -abs(float(route_len) - float(self.cfg.route_target_length_m))

    def _dedupe_waypoints(self, wps: Sequence[carla.Waypoint], min_sep: float = 0.75) -> List[carla.Waypoint]:
        out: List[carla.Waypoint] = []
        prev: Optional[carla.Waypoint] = None
        for wp in wps:
            if prev is None or distance2d(prev.transform.location, wp.transform.location) > min_sep:
                out.append(wp)
                prev = wp
        return out

    def _compute_route_cumdist(self) -> None:
        self.route_cumdist = []
        s = 0.0
        for i, wp in enumerate(self.route_wps):
            if i == 0:
                self.route_cumdist.append(0.0)
            else:
                s += distance2d(self.route_wps[i - 1].transform.location, wp.transform.location)
                self.route_cumdist.append(float(s))
        self.route_total_len_m = float(self.route_cumdist[-1]) if self.route_cumdist else 0.0

    def _route_projection_global(self, loc: carla.Location) -> Tuple[float, float, int]:
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0
        p = np.array([loc.x, loc.y], dtype=np.float32)
        best_d = 1e9
        best_s = 0.0
        best_idx = 0
        for i in range(len(self.route_xy) - 1):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            _, d, local_s = point_segment_projection(p, a, b)
            if d < best_d:
                best_d = d
                best_s = self.route_cumdist[i] + local_s
                best_idx = i
        return float(best_s), float(best_d), int(best_idx)


    def _route_projection_monotonic(self, loc: carla.Location) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        if len(self.route_xy) < 2:
            return 0.0, 0.0, 0, np.zeros(2, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)
        p = np.array([loc.x, loc.y], dtype=np.float32)
        start_idx = max(0, self.route_progress_idx - self.cfg.route_search_back)
        end_idx = min(len(self.route_xy) - 2, self.route_progress_idx + self.cfg.route_search_ahead)
        best_d = 1e9
        best_s = 0.0
        best_idx = start_idx
        best_proj = np.zeros(2, dtype=np.float32)
        best_dir = np.array([1.0, 0.0], dtype=np.float32)
        for i in range(start_idx, end_idx + 1):
            a = np.array(self.route_xy[i], dtype=np.float32)
            b = np.array(self.route_xy[i + 1], dtype=np.float32)
            proj, d, local_s = point_segment_projection(p, a, b)
            if d < best_d:
                seg = b - a
                seg_norm = float(np.linalg.norm(seg))
                seg_dir = seg / max(seg_norm, 1e-6)
                best_d = d
                best_s = self.route_cumdist[i] + local_s
                best_idx = i
                best_proj = proj
                best_dir = seg_dir
        self.route_progress_idx = max(self.route_progress_idx, best_idx)
        return float(best_s), float(best_d), int(best_idx), best_proj, best_dir

    def _route_projection(self, loc: carla.Location) -> Tuple[float, float]:
        s_arc, dL, _, _, _ = self._route_projection_monotonic(loc)
        return float(s_arc), float(dL)

    def _route_reference_state(self, loc: carla.Location, lookahead_m: float) -> Dict[str, object]:
        s_arc, dL, _, _, _ = self._route_projection_monotonic(loc)
        remaining_s = max(0.0, self.route_total_len_m - s_arc)

        if len(self.route_wps) == 0:
            return {
                "s_arc": 0.0,
                "dL": 0.0,
                "remaining_s": 0.0,
                "ref_idx": 0,
                "ref_wp": None,
                "signed_lane_err": 0.0,
                "heading_err": 0.0,
                "lane_width": 3.5,
                "wrong_lane": False,
                "opposite_lane": False,
            }

        ref_s = min(s_arc + max(2.0, 0.5 * lookahead_m), self.route_total_len_m)
        ref_idx = int(np.searchsorted(np.asarray(self.route_cumdist, dtype=np.float32), ref_s, side="left"))
        ref_idx = int(np.clip(ref_idx, 0, len(self.route_wps) - 1))
        ref_wp = self.route_wps[ref_idx]
        ref_tf = ref_wp.transform
        ref_loc = ref_tf.location

        route_yaw = math.radians(ref_tf.rotation.yaw)
        ego_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        route_right = np.array([-math.sin(route_yaw), math.cos(route_yaw)], dtype=np.float32)
        rel = np.array([loc.x - ref_loc.x, loc.y - ref_loc.y], dtype=np.float32)
        signed_lane_err = float(np.dot(rel, route_right))
        heading_err = wrap_pi(route_yaw - ego_yaw)
        lane_width = max(float(getattr(ref_wp, "lane_width", 3.5)), 3.5)

        ego_wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        wrong_lane = False
        opposite_lane = False
        if ego_wp is not None and not ref_wp.is_junction:
            same_road = ego_wp.road_id == ref_wp.road_id and ego_wp.section_id == ref_wp.section_id
            same_lane = same_road and (ego_wp.lane_id == ref_wp.lane_id)
            if same_road and ego_wp.lane_id != 0 and ref_wp.lane_id != 0:
                opposite_lane = np.sign(ego_wp.lane_id) != np.sign(ref_wp.lane_id)
            wrong_lane = opposite_lane or ((not same_lane) and (abs(signed_lane_err) > 0.55 * lane_width))

        return {
            "s_arc": float(s_arc),
            "dL": float(dL),
            "remaining_s": float(remaining_s),
            "ref_idx": int(ref_idx),
            "ref_wp": ref_wp,
            "signed_lane_err": float(signed_lane_err),
            "heading_err": float(heading_err),
            "lane_width": float(lane_width),
            "wrong_lane": bool(wrong_lane),
            "opposite_lane": bool(opposite_lane),
        }

    def _build_route_polyline_local(self, start_wp: carla.Waypoint, ds: float = 2.0) -> carla.Waypoint:
        """Extend a waypoint chain until its arc-length reaches ~210 m.

        210 m gives a small buffer above the 200 m target so that the
        GlobalRoutePlanner trace (used for fixed-goal routes) can trim cleanly
        to the 200 m window defined by route_soft_min/max_length_m.
        """
        self.route_xy = []
        self.route_wps = []
        cur = start_wp
        dist = 0.0
        while dist < max(self.cfg.min_route_length_m, 210.0):
            loc = cur.transform.location
            self.route_xy.append((float(loc.x), float(loc.y)))
            self.route_wps.append(cur)
            nxt = cur.next(ds)
            if not nxt:
                break
            if len(nxt) == 1:
                cur = nxt[0]
            else:
                best = nxt[0]
                best_score = -1e9
                for cand in nxt:
                    score = 0.0
                    if cand.road_id == cur.road_id:
                        score += 2.0
                    if cand.lane_id == cur.lane_id:
                        score += 4.0
                    score -= abs(float(cand.lane_id - cur.lane_id)) * 0.5
                    if score > best_score:
                        best_score = score
                        best = cand
                cur = best
            dist += ds
        return cur if self.route_wps else start_wp

    def _project_drive_wp(self, loc: carla.Location) -> Optional[carla.Waypoint]:
        try:
            return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        except BaseException:
            return None

    def _snap_spawn_to_driving_lane(self, tf: carla.Transform) -> carla.Transform:
        wp = self._project_drive_wp(tf.location)
        if wp is None:
            return carla.Transform(
                carla.Location(tf.location.x, tf.location.y, tf.location.z),
                carla.Rotation(tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
            )
        lane_tf = wp.transform
        z = max(float(tf.location.z), float(lane_tf.location.z)) + max(0.15, float(self.cfg.spawn_retry_lift_m))
        return carla.Transform(
            carla.Location(float(lane_tf.location.x), float(lane_tf.location.y), z),
            carla.Rotation(float(lane_tf.rotation.pitch), float(lane_tf.rotation.yaw), float(lane_tf.rotation.roll)),
        )

    def _build_route_from_goal_index(self, start_tf: carla.Transform, start_wp: carla.Waypoint, goal_index: int) -> bool:
        grp = self._get_global_planner()
        if grp is None:
            return False
        if goal_index < 0 or goal_index >= len(self.spawn_points):
            return False
        goal_sp = self.spawn_points[goal_index]

        start_wp_proj = self._project_drive_wp(start_tf.location) or start_wp
        goal_wp_proj = self._project_drive_wp(goal_sp.location)
        if goal_wp_proj is None:
            return False

        start_loc = start_wp_proj.transform.location
        goal_loc_req = goal_wp_proj.transform.location
        if distance2d(start_loc, goal_loc_req) < self.cfg.candidate_goal_min_dist_m:
            return False
        try:
            trace = grp.trace_route(start_loc, goal_loc_req)
        except BaseException:
            return False
        if len(trace) < 2:
            return False
        route = [wp for wp, _ in trace if wp is not None]
        route = self._dedupe_waypoints(route, min_sep=0.75)
        route_len = self._route_length_from_wps(route)
        if len(route) < 2 or route_len < self.cfg.min_route_length_m:
            return False
        if self.cfg.enforce_route_length_for_fixed_goal and not self._route_length_ok(route_len, strict=False):
            return False
        self.route_wps = route
        self.route_xy = [(float(wp.transform.location.x), float(wp.transform.location.y)) for wp in route]
        self._compute_route_cumdist()
        goal_loc = route[-1].transform.location
        goal_rot = route[-1].transform.rotation
        self.goal_transform = carla.Transform(carla.Location(goal_loc.x, goal_loc.y, goal_loc.z + 0.5), goal_rot)
        self.goal_spawn_transform = goal_sp
        self.current_goal_index = goal_index
        return True

    def _find_best_auto_route_from_spawn(
        self,
        start_tf: carla.Transform,
        start_wp: carla.Waypoint,
    ) -> Tuple[List[carla.Waypoint], Optional[int], float]:
        start_loc = start_wp.transform.location
        start_yaw = math.radians(start_tf.rotation.yaw)
        start_dir = np.array([math.cos(start_yaw), math.sin(start_yaw)], dtype=np.float32)
        grp = self._get_global_planner()
        if grp is None:
            return [], None, 0.0

        candidate_ids = list(range(len(self.spawn_points)))
        random.shuffle(candidate_ids)
        max_tries = min(len(candidate_ids), max(self.cfg.candidate_goal_max_tries, len(candidate_ids)))

        best_soft_route: List[carla.Waypoint] = []
        best_soft_goal_idx: Optional[int] = None
        best_soft_score = -1e18

        best_relaxed_route: List[carla.Waypoint] = []
        best_relaxed_goal_idx: Optional[int] = None
        best_relaxed_score = -1e18

        best_any_route: List[carla.Waypoint] = []
        best_any_goal_idx: Optional[int] = None
        best_any_score = -1e18

        for idx in candidate_ids[:max_tries]:
            if self.current_spawn_index is not None and idx == int(self.current_spawn_index):
                continue
            goal_sp = self.spawn_points[idx]
            goal_wp_proj = self._project_drive_wp(goal_sp.location)
            if goal_wp_proj is None:
                continue
            goal_loc_req = goal_wp_proj.transform.location
            euclid = distance2d(start_loc, goal_loc_req)
            if euclid < self.cfg.candidate_goal_min_dist_m:
                continue
            try:
                trace = grp.trace_route(start_loc, goal_loc_req)
            except BaseException:
                continue
            if len(trace) < 2:
                continue
            route = [wp for wp, _ in trace if wp is not None]
            route = self._dedupe_waypoints(route, min_sep=0.75)
            if len(route) < 2:
                continue
            route_len = self._route_length_from_wps(route)
            if route_len < self.cfg.min_route_length_m:
                continue

            detour_ratio = route_len / max(euclid, 1e-3)
            if detour_ratio > 3.5:
                continue

            k = min(3, len(route) - 1)
            p0 = np.array([route[0].transform.location.x, route[0].transform.location.y], dtype=np.float32)
            pk = np.array([route[k].transform.location.x, route[k].transform.location.y], dtype=np.float32)
            tangent = pk - p0
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-6:
                continue
            tangent = tangent / tangent_norm
            align = float(np.dot(start_dir, tangent))
            if align < -0.35:
                continue

            len_error = abs(route_len - self.cfg.route_target_length_m)
            score = (
                16.0 * align
                + 0.08 * euclid
                - 2.25 * len_error
                - 6.0 * max(detour_ratio - 1.8, 0.0)
            )
            any_score = 12.0 * align + 0.15 * route_len + 1.20 * euclid - 4.0 * max(detour_ratio - 2.2, 0.0)

            if self._route_length_ok(route_len, strict=False):
                if score > best_soft_score:
                    best_soft_score = score
                    best_soft_route = route
                    best_soft_goal_idx = idx
            elif route_len >= self.cfg.min_route_length_m and route_len <= max(self.cfg.route_target_length_m + 180.0, self.cfg.route_soft_max_length_m + 120.0):
                relaxed_score = score - 0.010 * max(0.0, len_error - self.cfg.route_target_tolerance_m)
                if relaxed_score > best_relaxed_score:
                    best_relaxed_score = relaxed_score
                    best_relaxed_route = route
                    best_relaxed_goal_idx = idx

            if any_score > best_any_score:
                best_any_score = any_score
                best_any_route = route
                best_any_goal_idx = idx

        if len(best_soft_route) >= 2:
            return best_soft_route, best_soft_goal_idx, self._route_length_from_wps(best_soft_route)
        if len(best_relaxed_route) >= 2:
            return best_relaxed_route, best_relaxed_goal_idx, self._route_length_from_wps(best_relaxed_route)
        if len(best_any_route) >= 2:
            return best_any_route, best_any_goal_idx, self._route_length_from_wps(best_any_route)
        return [], None, 0.0

    def _build_route(self, start_tf: carla.Transform, start_wp: carla.Waypoint, goal_index: Optional[int] = None) -> None:
        self.route_xy = []
        self.route_wps = []
        self.route_cumdist = []
        self.goal_transform = None
        self.goal_spawn_transform = None
        self.current_goal_index = None
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        chosen_goal = goal_index
        if chosen_goal is None and self.cfg.use_fixed_destination:
            chosen_goal = self.fixed_goal_index
        if chosen_goal is not None and int(chosen_goal) < 0:
            chosen_goal = None

        if chosen_goal is not None:
            if self._build_route_from_goal_index(start_tf, start_wp, int(chosen_goal)):
                return
            if self.cfg.strict_goal_route and not self.cfg.allow_fallback_route:
                raise RuntimeError(
                    f"Requested goal {chosen_goal} is not reachable from spawn {self.current_spawn_index}. "
                    "Choose another goal or use -1 for auto route selection."
                )

        best_route, best_goal_idx, best_route_len = self._find_best_auto_route_from_spawn(start_tf, start_wp)
        if len(best_route) >= 2:
            self.route_wps = best_route
            self.route_xy = [(float(wp.transform.location.x), float(wp.transform.location.y)) for wp in best_route]
            self._compute_route_cumdist()
            goal_loc = best_route[-1].transform.location
            goal_rot = best_route[-1].transform.rotation
            self.goal_transform = carla.Transform(carla.Location(goal_loc.x, goal_loc.y, goal_loc.z + 0.5), goal_rot)
            self.current_goal_index = best_goal_idx
            self.goal_spawn_transform = self.spawn_points[best_goal_idx] if best_goal_idx is not None else None
            if self.cfg.debug_mode and (not self._route_length_ok(best_route_len, strict=False)):
                print(
                    f"[WARN] auto-route length relaxed: chosen route_len={best_route_len:.1f}m "
                    f"for spawn {self.current_spawn_index} in {self.town_name}"
                )
            return

        last_wp = self._build_route_polyline_local(start_wp, ds=self.cfg.route_point_spacing_m)
        self._compute_route_cumdist()
        goal_loc = last_wp.transform.location
        goal_rot = last_wp.transform.rotation
        self.goal_transform = carla.Transform(carla.Location(goal_loc.x, goal_loc.y, goal_loc.z + 0.5), goal_rot)
        self.current_goal_index = None
        self.goal_spawn_transform = None
        if self.cfg.debug_mode:
            print(
                f"[WARN] planner auto-route unavailable from spawn {self.current_spawn_index} in {self.town_name}; "
                "using local lane-follow route fallback"
            )

    def _route_completion_pct(self, s_arc: Optional[float] = None) -> float:
        if self.route_total_len_m <= 1e-6:
            return 0.0
        if s_arc is None and self.vehicle is not None:
            s_arc, _ = self._route_projection(self.vehicle.get_location())
        s_arc = 0.0 if s_arc is None else float(s_arc)
        return float(np.clip(100.0 * s_arc / max(self.route_total_len_m, 1e-6), 0.0, 100.0))

    def _curvature_ahead(self, wp: Optional[carla.Waypoint], ds: float = 4.0) -> float:
        if wp is None:
            return 0.0
        nxt = wp.next(ds)
        if not nxt:
            return 0.0
        nxt2 = nxt[0].next(ds)
        if not nxt2:
            return 0.0
        p0 = np.array([wp.transform.location.x, wp.transform.location.y], dtype=np.float32)
        p1 = np.array([nxt[0].transform.location.x, nxt[0].transform.location.y], dtype=np.float32)
        p2 = np.array([nxt2[0].transform.location.x, nxt2[0].transform.location.y], dtype=np.float32)
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if float(a * b * c) < 1e-6:
            return 0.0
        area2 = abs(np.cross(p1 - p0, p2 - p0))
        return float(2.0 * area2 / (a * b * c))

    def _lane_heading_offset(self, wp_onlane: Optional[carla.Waypoint]) -> float:
        if wp_onlane is None or self.vehicle is None:
            return 0.0
        yaw_lane = math.radians(wp_onlane.transform.rotation.yaw)
        yaw_veh = math.radians(self.vehicle.get_transform().rotation.yaw)
        return float(np.clip(wrap_pi(yaw_veh - yaw_lane) / (math.pi / 2.0), -1.0, 1.0))

    @staticmethod
    def _tl_state_name(state: object) -> str:
        try:
            return str(state).split(".")[-1]
        except BaseException:
            return "Unknown"

    def _get_active_traffic_light_info(self) -> Tuple[str, float]:
        if self.vehicle is None:
            return "None", 1e9
        loc = self.vehicle.get_location()
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_light = self.vehicle.get_traffic_light()
        if ego_light is None or wp_onlane is None:
            return "None", 1e9
        try:
            tl_state = self._tl_state_name(ego_light.get_state())
        except BaseException:
            tl_state = "Unknown"
        tl_dist = self.cfg.tl_near_dist + 100.0
        try:
            stop_wps = ego_light.get_stop_waypoints()
            yaw_lane = math.radians(wp_onlane.transform.rotation.yaw)
            lane_fwd = np.array([math.cos(yaw_lane), math.sin(yaw_lane)], dtype=np.float32)
            best = self.cfg.tl_near_dist + 100.0
            for swp in stop_wps or []:
                vec = swp.transform.location - loc
                vec2 = np.array([vec.x, vec.y], dtype=np.float32)
                d = float(np.linalg.norm(vec2))
                if d < 0.5:
                    continue
                dot = float(np.dot(vec2, lane_fwd) / (d + 1e-6))
                if dot > 0.7 and d < best:
                    best = d
            tl_dist = best
        except BaseException:
            pass
        return tl_state, float(tl_dist)

    def _world_actors(self) -> Iterable[carla.Actor]:
        try:
            return self.world.get_actors()
        except BaseException:
            return []

    def _nearby_walkers(self, ego_loc: carla.Location) -> List[carla.Actor]:
        walkers: List[carla.Actor] = []
        try:
            actors = self._world_actors()
            if hasattr(actors, "filter"):
                iterable = actors.filter("walker.pedestrian.*")
            else:
                iterable = [a for a in actors if "walker.pedestrian" in getattr(a, "type_id", "")]
            for actor in iterable:
                if distance2d(actor.get_location(), ego_loc) <= self.cfg.entity_max_dist:
                    walkers.append(actor)
        except Exception:
            pass
        return walkers

    def _get_front_vehicle_info(self, max_dist: Optional[float] = None) -> Tuple[Optional[float], float, Optional[carla.Vehicle]]:
        if self.vehicle is None:
            return None, 0.0, None

        if max_dist is None:
            max_dist = self.cfg.front_vehicle_max_dist

        ego_loc = self.vehicle.get_location()
        ego_speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        lookahead_m = max(self.cfg.lookahead_base_m + 0.25 * max(ego_speed_kmh, 0.0), 5.0)

        rs = self._route_reference_state(ego_loc, lookahead_m=lookahead_m)
        ref_wp = rs.get("ref_wp", None)
        if ref_wp is None:
            return None, 0.0, None

        ref_loc = ref_wp.transform.location
        route_yaw = math.radians(ref_wp.transform.rotation.yaw)
        route_fwd = np.array([math.cos(route_yaw), math.sin(route_yaw)], dtype=np.float32)
        route_right = np.array([-math.sin(route_yaw), math.cos(route_yaw)], dtype=np.float32)
        lane_width = max(float(getattr(ref_wp, "lane_width", 3.5)), 3.5)

        best_metric = 1e9
        best_dist: Optional[float] = None
        best_speed_kmh = 0.0
        best_actor: Optional[carla.Vehicle] = None

        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            try:
                nloc = npc.get_location()
                rel_ego = np.array([nloc.x - ego_loc.x, nloc.y - ego_loc.y], dtype=np.float32)
                rel_ref = np.array([nloc.x - ref_loc.x, nloc.y - ref_loc.y], dtype=np.float32)
                forward_dist = float(np.dot(rel_ego, route_fwd))
                lateral_dist = abs(float(np.dot(rel_ref, route_right)))

                if forward_dist <= 0.5 or forward_dist > float(max_dist):
                    continue
                if lateral_dist > max(1.40, 0.45 * lane_width):
                    continue

                nwp = self.map.get_waypoint(nloc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if nwp is None:
                    continue
                npc_yaw = math.radians(nwp.transform.rotation.yaw)
                yaw_err = abs(wrap_pi(npc_yaw - route_yaw))
                same_road_family = (nwp.road_id == ref_wp.road_id) or bool(ref_wp.is_junction) or bool(nwp.is_junction)
                same_direction = yaw_err < 0.60
                if not same_road_family or not same_direction:
                    continue

                metric = forward_dist + 1.75 * lateral_dist
                if metric < best_metric:
                    best_metric = metric
                    best_dist = forward_dist
                    best_actor = npc
                    best_speed_kmh = 3.6 * vec3_length(npc.get_velocity())
            except Exception:
                continue

        return best_dist, float(best_speed_kmh), best_actor

    def _get_min_vehicle_ttc(self, max_dist: float = 25.0) -> Tuple[float, float]:
        if self.vehicle is None:
            return 999.0, 1e9
        ego_loc = self.vehicle.get_location()
        ego_vel = self.vehicle.get_velocity()
        ego_v = np.array([ego_vel.x, ego_vel.y], dtype=np.float32)
        min_ttc = 999.0
        min_dist = 1e9
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            try:
                nloc = npc.get_location()
                dist = distance2d(ego_loc, nloc)
                if dist < 0.5 or dist > max_dist:
                    continue
                min_dist = min(min_dist, dist)
                nvel = npc.get_velocity()
                rel_p = np.array([nloc.x - ego_loc.x, nloc.y - ego_loc.y], dtype=np.float32)
                rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_v
                closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
                if closing > 1e-3:
                    min_ttc = min(min_ttc, float(dist / closing))
            except Exception:
                continue
        return float(min_ttc), float(min_dist)

    def _observation_dropout(self, dist: float, fog_norm: float, kind: str = "vehicle") -> bool:
        dist_n = float(np.clip(dist / max(self.cfg.entity_max_dist, 1e-3), 0.0, 1.0))
        miss = self.cfg.obs_miss_base + 0.22 * dist_n + 0.22 * fog_norm
        if kind == "walker":
            miss += 0.08
        if kind == "traffic_light":
            miss += 0.05
        miss = float(np.clip(miss, 0.02, 0.65))
        return random.random() < miss

    def _noisy_relative_measurement(
        self,
        delta_world: np.ndarray,
        rel_v_world: np.ndarray,
        dist: float,
        fog_norm: float,
        ego_forward: np.ndarray,
        ego_left: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        dist_n = float(np.clip(dist / max(self.cfg.entity_max_dist, 1e-3), 0.0, 1.0))
        pos_std = self.cfg.obs_pos_noise_m * (0.45 + 0.80 * dist_n + 0.90 * fog_norm)
        vel_std = self.cfg.obs_vel_noise_ms * (0.35 + 0.75 * dist_n + 0.80 * fog_norm)
        noisy_delta = delta_world + np.random.normal(0.0, pos_std, size=2).astype(np.float32)
        noisy_rel_v = rel_v_world + np.random.normal(0.0, vel_std, size=2).astype(np.float32)
        rel_x = float(np.dot(noisy_delta, ego_forward) / self.cfg.entity_max_dist)
        rel_y = float(np.dot(noisy_delta, ego_left) / self.cfg.entity_max_dist)
        rel_vx = float(np.dot(noisy_rel_v, ego_forward) / 15.0)
        rel_vy = float(np.dot(noisy_rel_v, ego_left) / 10.0)
        sigma2 = float(np.clip(0.05 + 0.22 * dist_n + 0.28 * fog_norm + 0.08 * pos_std + 0.04 * vel_std, 0.05, 0.95))
        return rel_x, rel_y, rel_vx, rel_vy, sigma2

    def _collect_entity_features(self) -> Tuple[np.ndarray, np.ndarray, float]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        curv = self._curvature_ahead(wp_onlane)
        lane_curv = float(np.tanh(10.0 * curv))
        lane_head = float(self._lane_heading_offset(wp_onlane))
        ego_tf = self.vehicle.get_transform()
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
        ego_left = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)
        fog_norm = get_fog_norm(self.world)

        feats: List[Tuple[float, np.ndarray]] = []
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            nloc = npc.get_location()
            delta = np.array([nloc.x - loc.x, nloc.y - loc.y], dtype=np.float32)
            dist = float(np.linalg.norm(delta))
            if dist < 1.5 or dist > self.cfg.entity_max_dist:
                continue
            if self._observation_dropout(dist, fog_norm, kind="vehicle"):
                continue
            nvel = npc.get_velocity()
            rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_vel
            rel_x, rel_y, rel_vx, rel_vy, sigma2 = self._noisy_relative_measurement(delta, rel_v, dist, fog_norm, ego_forward, ego_left)
            c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            kappa = np.array([lane_curv, lane_head], dtype=np.float32)
            e = np.concatenate([np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
            feats.append((dist, e))

        for walker in self._nearby_walkers(loc):
            try:
                wloc = walker.get_location()
                delta = np.array([wloc.x - loc.x, wloc.y - loc.y], dtype=np.float32)
                dist = float(np.linalg.norm(delta))
                if dist < 1.5 or dist > self.cfg.entity_max_dist:
                    continue
                if self._observation_dropout(dist, fog_norm, kind="walker"):
                    continue
                wvel = walker.get_velocity()
                rel_v = np.array([wvel.x, wvel.y], dtype=np.float32) - ego_vel
                rel_x, rel_y, rel_vx, rel_vy, sigma2 = self._noisy_relative_measurement(delta, rel_v, dist, fog_norm, ego_forward, ego_left)
                sigma2 = float(np.clip(sigma2 + 0.06, 0.05, 0.98))
                c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                kappa = np.array([lane_curv, lane_head], dtype=np.float32)
                e = np.concatenate([np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
                feats.append((dist, e))
            except Exception:
                continue

        tl_state, tl_dist = self._get_active_traffic_light_info()
        if tl_dist < self.cfg.tl_near_dist and (not self._observation_dropout(tl_dist, fog_norm, kind="traffic_light")):
            noisy_tl_dist = float(max(0.0, tl_dist + np.random.normal(0.0, self.cfg.tl_obs_noise_m * (0.35 + fog_norm))))
            rel_x = float(np.clip(noisy_tl_dist / self.cfg.entity_max_dist, 0.0, 1.0))
            sigma2 = float(np.clip(0.08 + 0.24 * (noisy_tl_dist / max(self.cfg.tl_near_dist, 1e-3)) + 0.25 * fog_norm, 0.05, 0.90))
            c = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            kappa = np.array([lane_curv, lane_head], dtype=np.float32)
            e = np.concatenate([np.array([rel_x, 0.0, 0.0, 0.0], dtype=np.float32), c, kappa, np.array([sigma2], dtype=np.float32)])
            feats.append((noisy_tl_dist, e))

        feats.sort(key=lambda x: x[0])
        feats = feats[: self.cfg.max_entities]

        edges = np.zeros((self.cfg.max_entities, self.cfg.edge_dim), dtype=np.float32)
        mask = np.zeros((self.cfg.max_entities,), dtype=np.float32)
        if len(feats) == 0:
            mask[0] = 1.0
            edges[0, -1] = 0.8
            nearest = self.cfg.entity_max_dist
        else:
            for i, (_, e) in enumerate(feats):
                edges[i, :] = e
                mask[i] = 1.0
            nearest = float(feats[0][0])
        return edges, mask, nearest

    def _route_observation_features(self) -> Dict[str, float]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        ego_speed_ms = vec3_length(vel)
        ego_speed_kmh = 3.6 * ego_speed_ms
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_curv = float(np.tanh(10.0 * self._curvature_ahead(wp_onlane)))
        lane_head = float(self._lane_heading_offset(wp_onlane))

        lookahead_m = max(self.cfg.lookahead_base_m + self.cfg.lookahead_speed_gain * max(ego_speed_kmh, 0.0), 4.0)
        rs = self._route_reference_state(loc, lookahead_m=lookahead_m)
        lane_width = max(float(rs.get('lane_width', 3.5)), 3.5)
        signed_lane_err = float(rs.get('signed_lane_err', 0.0))
        heading_err = float(rs.get('heading_err', 0.0))
        ref_wp = rs.get('ref_wp', None)

        ego_tf = self.vehicle.get_transform()
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
        ego_left = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)

        look_x = 0.0
        look_y = 0.0
        v_route = 0.0
        if ref_wp is not None:
            ref_loc = ref_wp.transform.location
            delta = np.array([ref_loc.x - loc.x, ref_loc.y - loc.y], dtype=np.float32)
            look_x = float(np.clip(np.dot(delta, ego_forward) / max(lookahead_m, 1.0), -2.0, 2.0))
            look_y = float(np.clip(np.dot(delta, ego_left) / max(1.5 * lane_width, 1.0), -2.0, 2.0))
            route_yaw = math.radians(ref_wp.transform.rotation.yaw)
            route_fwd = np.array([math.cos(route_yaw), math.sin(route_yaw)], dtype=np.float32)
            v_route = float(np.clip(np.dot(ego_vel, route_fwd) / 10.0, -1.5, 2.0))

        fog_norm = get_fog_norm(self.world)
        density_norm = float(np.clip(len(self.npcs) / max(self.cfg.npc_max, 1), 0.0, 1.0))
        curv_norm = float(np.clip(abs(lane_curv), 0.0, 1.0))
        muA = float(np.clip(1.0 - (0.45 * density_norm + 0.35 * fog_norm + 0.20 * curv_norm), 0.0, 1.0))

        return {
            'lane_curv': float(lane_curv),
            'lane_head': float(lane_head),
            'route_cte': float(np.clip(signed_lane_err / max(lane_width, 1.0), -2.0, 2.0)),
            'route_heading': float(np.clip(heading_err / (math.pi / 2.0), -1.0, 1.0)),
            'lookahead_x': float(look_x),
            'lookahead_y': float(look_y),
            'v_route': float(v_route),
            'wrong_lane': float(1.0 if bool(rs.get('wrong_lane', False)) else 0.0),
            'opposite_lane': float(1.0 if bool(rs.get('opposite_lane', False)) else 0.0),
            'muA': float(muA),
            'remaining_s': float(rs.get('remaining_s', 0.0)),
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed = vec3_length(vel)
        v_ego = float(np.clip(speed / 10.0, 0.0, 2.0))

        s_arc, _ = self._route_projection(loc)
        remaining_s = max(0.0, self.route_total_len_m - s_arc)
        total_route_len = max(self.route_total_len_m, 1.0)
        d_goal = float(np.clip(remaining_s / total_route_len, 0.0, 1.5))

        route_obs = self._route_observation_features()
        scalars = np.array(
            [
                v_ego,
                float(self.prev_action[0]),
                float(self.prev_action[1]),
                float(self.prev_action[2]),
                d_goal,
                float(route_obs['lane_curv']),
                float(route_obs['lane_head']),
                float(route_obs['route_cte']),
                float(route_obs['route_heading']),
                float(route_obs['lookahead_x']),
                float(route_obs['lookahead_y']),
                float(route_obs['v_route']),
                float(route_obs['muA']),
            ],
            dtype=np.float32,
        )
        edges, mask, _ = self._collect_entity_features()
        return {"scalars": scalars, "edges": edges, "mask": mask}

    def _compute_route_guidance_action(self, tl_state: str, tl_dist: float, front_vehicle_dist: Optional[float], front_vehicle_speed_kmh: float) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.vehicle is None or len(self.route_wps) < 2:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), {
                "desired_speed_kmh": 0.0,
                "route_idx": 0.0,
                "route_target_idx": 0.0,
                "route_cte": 0.0,
                "route_heading_err": 0.0,
                "route_remaining_s": 0.0,
                "route_dL": 0.0,
                "wrong_lane": 0.0,
                "opposite_lane": 0.0,
            }

        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed_kmh = 3.6 * vec3_length(vel)
        lookahead_m = max(self.cfg.lookahead_base_m + self.cfg.lookahead_speed_gain * max(speed_kmh, 0.0), 4.0)
        rs = self._route_reference_state(loc, lookahead_m=lookahead_m)
        dL = float(rs["dL"])
        remaining_s = float(rs["remaining_s"])
        ref_idx = int(rs["ref_idx"])
        ref_wp = rs["ref_wp"]
        signed_lane_err = float(rs["signed_lane_err"])
        heading_err = float(rs["heading_err"])
        lane_width = float(rs["lane_width"])
        wrong_lane = bool(rs["wrong_lane"])
        opposite_lane = bool(rs["opposite_lane"])

        curv = self._curvature_ahead(ref_wp)
        curv_n = abs(float(np.tanh(10.0 * curv)))
        desired_speed_kmh = self.cfg.target_speed_kmh - self.cfg.curve_speed_penalty_kmh * curv_n
        desired_speed_kmh = max(self.cfg.min_target_speed_kmh, desired_speed_kmh)
        desired_speed_kmh = min(desired_speed_kmh, self.cfg.hard_speed_cap_kmh)

        if remaining_s < 30.0:
            desired_speed_kmh = min(desired_speed_kmh, 12.0)
        if remaining_s < 15.0:
            desired_speed_kmh = min(desired_speed_kmh, 8.0)
        if remaining_s < 8.0:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        abs_lane_err = abs(signed_lane_err)
        if abs_lane_err > 0.22 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 13.0)
        if abs_lane_err > 0.35 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 11.0)
        if abs_lane_err > 0.50 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 9.0)
        if abs_lane_err > 0.70 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 7.0)
        if abs_lane_err > 0.95 * lane_width:
            desired_speed_kmh = min(desired_speed_kmh, 5.5)

        if abs(heading_err) > 0.25:
            desired_speed_kmh = min(desired_speed_kmh, 11.0)
        if abs(heading_err) > 0.40:
            desired_speed_kmh = min(desired_speed_kmh, 8.5)
        if abs(heading_err) > 0.60:
            desired_speed_kmh = min(desired_speed_kmh, 6.5)

        if wrong_lane:
            desired_speed_kmh = min(desired_speed_kmh, 4.5)
        if opposite_lane:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        if tl_state == "Red":
            if tl_dist <= self.cfg.tl_stop_dist:
                desired_speed_kmh = 0.0
            elif tl_dist < self.cfg.tl_near_dist:
                ratio = (tl_dist - self.cfg.tl_stop_dist) / max(self.cfg.tl_near_dist - self.cfg.tl_stop_dist, 1e-3)
                desired_speed_kmh = min(desired_speed_kmh, max(0.0, ratio * self.cfg.min_target_speed_kmh))
        elif tl_state == "Yellow" and tl_dist < self.cfg.tl_stop_dist:
            desired_speed_kmh = min(desired_speed_kmh, 4.0)

        if front_vehicle_dist is not None:
            if front_vehicle_dist < self.cfg.front_vehicle_block_dist:
                desired_speed_kmh = 0.0
            elif front_vehicle_dist < self.cfg.front_vehicle_soft_block_dist:
                ratio = (front_vehicle_dist - self.cfg.front_vehicle_block_dist) / max(self.cfg.front_vehicle_soft_block_dist - self.cfg.front_vehicle_block_dist, 1e-3)
                desired_speed_kmh = min(desired_speed_kmh, max(0.0, ratio * max(front_vehicle_speed_kmh, 5.0)))
            elif front_vehicle_dist < self.cfg.front_vehicle_max_dist:
                desired_speed_kmh = min(desired_speed_kmh, max(front_vehicle_speed_kmh + 3.0, 7.0))

        cte_term = math.atan2(1.45 * signed_lane_err, max(0.45 * lookahead_m, 1.0))
        steer = (self.cfg.route_steer_kp + 0.55) * heading_err - (self.cfg.route_cte_kp + 0.55) * cte_term

        center_push = 0.0
        if abs_lane_err > self.cfg.center_deadband_m:
            lane_push_mag = max(0.0, abs_lane_err - self.cfg.center_push_start_m)
            center_push = self.cfg.center_push_gain * math.tanh(lane_push_mag / 0.75) * float(np.sign(signed_lane_err))
            steer -= center_push
            if abs_lane_err > self.cfg.route_hard_dL_m_tight:
                hard_push = self.cfg.center_push_hard_gain * math.tanh((abs_lane_err - self.cfg.route_hard_dL_m_tight) / 0.60)
                steer -= hard_push * float(np.sign(signed_lane_err))

        if wrong_lane:
            steer *= 1.20
        if opposite_lane:
            steer *= 1.35
        steer = float(np.clip(steer, -1.0, 1.0))
        if speed_kmh > 10.0:
            steer = float(np.clip(steer, -self.cfg.max_safe_steer_at_speed, self.cfg.max_safe_steer_at_speed))

        speed_err = desired_speed_kmh - speed_kmh
        if desired_speed_kmh <= 0.1:
            throttle = 0.0
            brake = float(np.clip(0.35 + (max(0.0, self.cfg.tl_stop_dist - tl_dist) / max(self.cfg.tl_stop_dist, 1e-3)), 0.35, 1.0))
        elif speed_err >= 0.0:
            throttle = float(np.clip(0.12 + self.cfg.route_speed_kp * speed_err, 0.0, 0.75))
            brake = 0.0
        elif speed_err > -self.cfg.coast_speed_band_kmh:
            throttle = 0.0
            brake = 0.0
        elif speed_err > -self.cfg.soft_brake_speed_excess_kmh:
            throttle = 0.0
            brake = float(np.clip(((-speed_err) - self.cfg.coast_speed_band_kmh) / 24.0, 0.0, 0.10))
        else:
            throttle = 0.0
            brake = float(np.clip(0.10 + (((-speed_err) - self.cfg.soft_brake_speed_excess_kmh) / 10.0), 0.0, 0.42))

        meta = {
            "desired_speed_kmh": float(desired_speed_kmh),
            "route_idx": float(ref_idx),
            "route_target_idx": float(min(ref_idx + 1, len(self.route_wps) - 1)),
            "route_cte": float(signed_lane_err),
            "route_heading_err": float(heading_err),
            "route_remaining_s": float(remaining_s),
            "route_dL": float(dL),
            "wrong_lane": float(1.0 if wrong_lane else 0.0),
            "opposite_lane": float(1.0 if opposite_lane else 0.0),
            "center_push": float(center_push),
        }
        return np.array([throttle, brake, steer], dtype=np.float32), meta

    def _policy_passthrough_filter(self, raw_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], bool]:
        assert self.vehicle is not None
        speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        tl_state, tl_dist = self._get_active_traffic_light_info()
        front_vehicle_dist, front_vehicle_speed_kmh, _ = self._get_front_vehicle_info()
        guidance, route_meta = self._compute_route_guidance_action(
            tl_state=tl_state, tl_dist=tl_dist,
            front_vehicle_dist=front_vehicle_dist, front_vehicle_speed_kmh=front_vehicle_speed_kmh)
        raw = np.asarray(raw_action, dtype=np.float32).copy()
        raw[0] = float(np.clip(raw[0], 0.0, 1.0))
        raw[1] = float(np.clip(raw[1], 0.0, 1.0))
        raw[2] = float(np.clip(raw[2], -1.0, 1.0))
        if raw[0] > 0.0 and raw[1] > 0.0:
            if raw[0] >= raw[1]: raw[1] = 0.0
            else: raw[0] = 0.0
        cte = abs(float(route_meta.get("route_cte", 0.0)))
        hdg = abs(float(route_meta.get("route_heading_err", 0.0)))
        wrong_lane = bool(route_meta.get("wrong_lane", 0.0) > 0.5 or route_meta.get("opposite_lane", 0.0) > 0.5)
        open_road = (tl_state != "Red") and (front_vehicle_dist is None or front_vehicle_dist > self.cfg.front_vehicle_soft_block_dist)
        route_hard = wrong_lane or cte > 1.40 or hdg > 0.28
        # Stuck detection
        if speed_kmh < self.cfg.stuck_speed_kmh and open_road and not route_hard:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            if tl_state == "Red" or (front_vehicle_dist is not None and front_vehicle_dist < self.cfg.front_vehicle_soft_block_dist):
                self.release_steps_left = 0
        if self.stuck_steps >= self.cfg.stuck_steps_threshold and open_road and not route_hard:
            self.release_steps_left = max(self.release_steps_left, self.cfg.release_duration_steps)
            self.stuck_steps = 0
        # Forced release: bypass rate limiter
        if self.release_steps_left > 0 and open_road and not route_hard:
            a = guidance.copy()
            a[0] = max(float(a[0]), self.cfg.release_throttle)
            a[1] = 0.0
            a[2] = float(np.clip(a[2], -self.cfg.low_speed_steer_limit, self.cfg.low_speed_steer_limit))
            self.release_steps_left -= 1
            route_meta["safety_tl_state"] = tl_state; route_meta["safety_tl_dist"] = float(tl_dist)
            route_meta["safety_front_vehicle_dist"] = -1.0 if front_vehicle_dist is None else float(front_vehicle_dist)
            route_meta["safety_front_vehicle_speed_kmh"] = float(front_vehicle_speed_kmh)
            route_meta["safety_front_ttc"] = 999.0; route_meta["safety_min_vehicle_ttc"] = 999.0
            route_meta["safety_reason"] = "unstuck"; route_meta["policy_route_blend"] = 1.0
            return np.asarray(a, dtype=np.float32), route_meta, True
        # Normal blend
        blend = float(self.cfg.policy_route_blend_base)
        if cte > 0.45 or hdg > 0.10: blend = max(blend, 0.55)
        if cte > 0.90 or hdg > 0.18: blend = max(blend, float(self.cfg.policy_route_blend_bad))
        if cte > 1.40 or hdg > 0.28 or wrong_lane: blend = max(blend, float(self.cfg.policy_route_blend_hard))
        if tl_state == "Red" and tl_dist < self.cfg.tl_near_dist: blend = max(blend, 0.78)
        if front_vehicle_dist is not None and front_vehicle_dist < (self.cfg.front_vehicle_soft_block_dist + 2.0): blend = max(blend, 0.78)
        steer_blend = max(0.60, blend)
        a = raw.copy()
        a[0] = (1.0 - blend) * float(a[0]) + blend * float(guidance[0])
        a[1] = (1.0 - blend) * float(a[1]) + blend * float(guidance[1])
        a[2] = (1.0 - steer_blend) * float(a[2]) + steer_blend * float(guidance[2])
        desired_speed = float(route_meta.get("desired_speed_kmh", speed_kmh))
        if open_road and speed_kmh < max(6.0, desired_speed - 2.0):
            a[0] = max(float(a[0]), min(0.42, float(guidance[0]) + 0.05))
            if a[1] < 0.12: a[1] = 0.0
        a[0] = float(np.clip(a[0], self.prev_action[0] - self.cfg.max_throttle_step, self.prev_action[0] + self.cfg.max_throttle_step))
        a[1] = float(np.clip(a[1], self.prev_action[1] - self.cfg.max_brake_step, self.prev_action[1] + self.cfg.max_brake_step))
        a[2] = float(np.clip(a[2], self.prev_action[2] - self.cfg.max_steer_step, self.prev_action[2] + self.cfg.max_steer_step))
        a[0] = float(np.clip(a[0], 0.0, 1.0)); a[1] = float(np.clip(a[1], 0.0, 1.0)); a[2] = float(np.clip(a[2], -1.0, 1.0))
        if a[0] > 0.0 and a[1] > 0.0:
            if a[0] >= a[1]: a[1] = 0.0
            else: a[0] = 0.0
        route_meta["safety_tl_state"] = tl_state; route_meta["safety_tl_dist"] = float(tl_dist)
        route_meta["safety_front_vehicle_dist"] = -1.0 if front_vehicle_dist is None else float(front_vehicle_dist)
        route_meta["safety_front_vehicle_speed_kmh"] = float(front_vehicle_speed_kmh)
        route_meta["safety_front_ttc"] = 999.0; route_meta["safety_min_vehicle_ttc"] = 999.0
        route_meta["safety_reason"] = "policy_route_assist"
        route_meta["guidance_steer"] = float(guidance[2])
        route_meta["policy_route_blend"] = float(blend)
        return np.asarray(a, dtype=np.float32), route_meta, False
        
    def _safety_filter(self, raw_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], bool]:
        assert self.vehicle is not None

        speed_kmh = 3.6 * vec3_length(self.vehicle.get_velocity())
        tl_state, tl_dist = self._get_active_traffic_light_info()
        front_vehicle_dist, front_vehicle_speed_kmh, _ = self._get_front_vehicle_info()

        guidance, route_meta = self._compute_route_guidance_action(
            tl_state=tl_state,
            tl_dist=tl_dist,
            front_vehicle_dist=front_vehicle_dist,
            front_vehicle_speed_kmh=front_vehicle_speed_kmh,
        )

        cte = abs(float(route_meta.get("route_cte", 0.0)))
        hdg = abs(float(route_meta.get("route_heading_err", 0.0)))
        wrong_lane = bool(route_meta.get("wrong_lane", 0.0) > 0.5)
        opposite_lane = bool(route_meta.get("opposite_lane", 0.0) > 0.5)

        route_bad = wrong_lane or opposite_lane or cte > min(self.cfg.route_override_dL_m, self.cfg.route_soft_dL_m) or hdg > self.cfg.route_override_heading_rad
        route_hard = wrong_lane or opposite_lane or cte > min(self.cfg.route_hard_dL_m, self.cfg.route_hard_dL_m_tight) or hdg > self.cfg.route_hard_heading_rad

        red_stop = tl_state == "Red" and tl_dist <= (self.cfg.tl_stop_dist + 0.5)

        ttc = 999.0
        front_blocked = False
        ego_speed_ms = vec3_length(self.vehicle.get_velocity())
        if front_vehicle_dist is not None:
            closing_ms = max(0.0, ego_speed_ms - (front_vehicle_speed_kmh / 3.6))
            if closing_ms > 1e-3:
                ttc = float(front_vehicle_dist / closing_ms)
            min_hard_dist = max(3.0, 0.8 + 0.25 * ego_speed_ms)
            front_blocked = front_vehicle_dist < min_hard_dist or (ttc < self.cfg.min_ttc_s and front_vehicle_dist < self.cfg.front_vehicle_soft_block_dist)

        min_vehicle_ttc, min_vehicle_dist = self._get_min_vehicle_ttc(max_dist=max(self.cfg.front_vehicle_max_dist, 25.0))
        ttc_caution = min_vehicle_ttc < self.cfg.caution_ttc_s and min_vehicle_dist < 18.0

        open_road = (not red_stop) and (front_vehicle_dist is None or front_vehicle_dist > max(12.0, self.cfg.front_vehicle_soft_block_dist + 1.0))

        if speed_kmh < self.cfg.stuck_speed_kmh and open_road and (not route_hard):
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            if red_stop or front_blocked:
                self.release_steps_left = 0

        if self.stuck_steps >= self.cfg.stuck_steps_threshold and open_road and (not route_hard):
            self.release_steps_left = max(self.release_steps_left, self.cfg.release_duration_steps)
            self.stuck_steps = 0

        desired_speed_now = float(route_meta.get("desired_speed_kmh", speed_kmh))
        alpha = 1.00 if route_hard else (0.90 if route_bad else 0.48)
        a = (1.0 - alpha) * raw_action + alpha * guidance
        intervention = bool(alpha >= 0.85 or route_bad or ttc_caution)
        intervention_reason = "route_blend" if intervention else "none"

        if open_road and speed_kmh < min(7.0, max(5.0, desired_speed_now)) and (not route_hard):
            a[0] = max(float(a[0]), min(0.34, float(guidance[0]) + 0.10))
            if a[1] < 0.28:
                a[1] = 0.0
        elif open_road and speed_kmh < max(10.0, desired_speed_now - 4.0) and (not route_hard):
            a[0] = max(float(a[0]), min(0.44, float(guidance[0]) + 0.08))
            if a[1] < 0.18:
                a[1] = 0.0

        if (not red_stop) and (not front_blocked):
            blend = self.cfg.steer_guidance_blend
            if route_bad:
                blend = max(blend, 0.82)
            elif cte > 0.55 or hdg > 0.16:
                blend = max(blend, 0.70)
            a[2] = blend * float(guidance[2]) + (1.0 - blend) * float(a[2])

        if (not red_stop) and (not front_blocked) and (not route_hard):
            soft_brake_limit = self.cfg.open_road_brake_curve_limit if (cte > 0.80 or hdg > 0.20 or ttc_caution) else self.cfg.open_road_brake_soft_limit
            if a[1] <= self.cfg.brake_release_threshold and speed_kmh <= desired_speed_now + self.cfg.coast_speed_band_kmh:
                a[1] = 0.0
                if speed_kmh < desired_speed_now - 0.5:
                    a[0] = max(float(a[0]), min(0.34, float(guidance[0]) + 0.03))
            elif a[1] > soft_brake_limit and speed_kmh <= desired_speed_now + self.cfg.soft_brake_speed_excess_kmh:
                a[1] = soft_brake_limit
                if speed_kmh < desired_speed_now - 1.0:
                    a[0] = max(float(a[0]), min(0.30, float(guidance[0]) + 0.04))

        if self.release_steps_left > 0 and open_road and (not route_hard):
            a = guidance.copy()
            a[0] = max(float(a[0]), self.cfg.release_throttle)
            a[1] = 0.0
            a[2] = float(np.clip(a[2], -self.cfg.low_speed_steer_limit, self.cfg.low_speed_steer_limit))
            self.release_steps_left -= 1
            intervention = True
            intervention_reason = "unstuck"

        if ttc_caution and (not red_stop):
            a[0] = min(float(a[0]), 0.22)
            if min_vehicle_ttc < self.cfg.hard_ttc_s:
                a[1] = max(float(a[1]), 0.18)
                intervention = True
                intervention_reason = "ttc_caution"

        if speed_kmh > self.cfg.hard_speed_cap_kmh:
            excess = speed_kmh - self.cfg.hard_speed_cap_kmh
            a[0] = 0.0
            a[1] = max(float(a[1]), float(np.clip(0.20 + 0.10 * excess, 0.20, 1.0)))
            a[2] = float(np.clip(guidance[2], -self.cfg.max_safe_steer_at_speed, self.cfg.max_safe_steer_at_speed))
            intervention = True
            intervention_reason = "speed_cap"

        if route_hard:
            a[0] = min(float(a[0]), 0.22)
            if speed_kmh > 9.0:
                a[1] = max(float(a[1]), 0.22)
            elif speed_kmh < 5.0:
                a[1] = min(float(a[1]), 0.06)
                a[0] = max(float(a[0]), 0.24)
            a[2] = float(guidance[2])
            intervention = True
            intervention_reason = "route_recovery"

        if red_stop:
            a[0] = 0.0
            a[1] = max(float(a[1]), 0.75)
            a[2] = float(np.clip(guidance[2], -0.25, 0.25))
            intervention = True
            intervention_reason = "red_light"
        elif front_blocked:
            a[0] = 0.0
            a[1] = max(float(a[1]), 0.65)
            a[2] = float(np.clip(guidance[2], -0.30, 0.30))
            intervention = True
            intervention_reason = "front_vehicle"

        a[0] = float(np.clip(a[0], self.prev_action[0] - self.cfg.max_throttle_step, self.prev_action[0] + self.cfg.max_throttle_step))
        a[1] = float(np.clip(a[1], self.prev_action[1] - self.cfg.max_brake_step, self.prev_action[1] + self.cfg.max_brake_step))
        a[2] = float(np.clip(a[2], self.prev_action[2] - self.cfg.max_steer_step, self.prev_action[2] + self.cfg.max_steer_step))

        a[0] = float(np.clip(a[0], 0.0, 1.0))
        a[1] = float(np.clip(a[1], 0.0, 1.0))
        a[2] = float(np.clip(a[2], -1.0, 1.0))
        if a[0] > 0.0 and a[1] > 0.0:
            if a[0] >= a[1]:
                a[1] = 0.0
            else:
                a[0] = 0.0

        route_meta["safety_tl_state"] = tl_state
        route_meta["safety_tl_dist"] = float(tl_dist)
        route_meta["safety_front_vehicle_dist"] = -1.0 if front_vehicle_dist is None else float(front_vehicle_dist)
        route_meta["safety_front_vehicle_speed_kmh"] = float(front_vehicle_speed_kmh)
        route_meta["safety_front_ttc"] = float(ttc)
        route_meta["safety_min_vehicle_ttc"] = float(min_vehicle_ttc)
        route_meta["safety_reason"] = intervention_reason

        return np.array(a, dtype=np.float32), route_meta, intervention
        
    def _actor_is_alive(self, actor: Optional[carla.Actor]) -> bool:
        if actor is None:
            return False
        try:
            return bool(actor.is_alive)
        except BaseException:
            return False

    def _safe_set_autopilot(self, actor: Optional[carla.Actor], enabled: bool) -> None:
        if not self._actor_is_alive(actor):
            return
        try:
            actor.set_autopilot(bool(enabled), self.tm.get_port())
            return
        except TypeError:
            pass
        except BaseException:
            pass
        try:
            actor.set_autopilot(bool(enabled))
        except BaseException:
            pass

    def _safe_destroy_one(self, actor: Optional[carla.Actor], sleep_s: float = 0.0) -> None:
        if not self._actor_is_alive(actor):
            return
        try:
            actor.destroy()
        except RuntimeError as e:
            # Ignore the exact error we are fixing.
            if "destroyed actor" not in str(e).lower() and self.cfg.debug_mode:
                print(f"[WARN] destroy actor runtime error: {e}")
        except BaseException as e:
            if self.cfg.debug_mode:
                print(f"[WARN] destroy actor failed: {e}")
        if sleep_s > 0.0:
            try:
                time.sleep(sleep_s)
            except BaseException:
                pass

    def _drain_world_ticks(self, count: int, label: str, sleep_s: float = 0.03) -> None:
        for i in range(max(0, int(count))):
            self._safe_tick(label=f"{label}_{i+1}", raise_on_fail=False)
            if sleep_s > 0.0:
                try:
                    time.sleep(sleep_s)
                except BaseException:
                    pass

    def _on_collision_event(self, event: object) -> None:
        if self._teardown_in_progress or (not self._episode_live):
            return
        if not self._actor_is_alive(self.vehicle):
            return
        try:
            self.collision_events.append(event)
        except BaseException:
            pass

    def _on_lane_invasion_event(self, event: object) -> None:
        if self._teardown_in_progress or (not self._episode_live):
            return
        if not self._actor_is_alive(self.vehicle):
            return
        try:
            self.lane_invasion_events.append(event)
        except BaseException:
            pass

    def _begin_episode_teardown(self, reason: str = "") -> None:
        """
        Stop new callbacks and detach Traffic Manager ownership before reset().
        This is intentionally lightweight and idempotent.
        """
        if self._teardown_in_progress:
            return

        self._teardown_in_progress = True
        self._episode_live = False
        self._terminal_reason = str(reason or self._terminal_reason or "")

        # Stop sensors first so no new callbacks are queued.
        for sensor in list(self.sensor_list):
            self._stop_sensor_only(sensor)

        # Detach TM from NPCs before any future destroy.
        for npc in list(self.npcs):
            self._safe_set_autopilot(npc, False)

        # Freeze ego control.
        if self._actor_is_alive(self.vehicle):
            self._safe_set_autopilot(self.vehicle, False)
            try:
                self.vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False,
                    )
                )
            except BaseException:
                pass

        # One drain tick here helps clear queued callbacks before reset() starts.
       # self._drain_world_ticks(1, "terminal_teardown", sleep_s=0.02)    

    def _setup_collision(self) -> None:
        if not self.cfg.enable_collision_sensor or not self._actor_is_alive(self.vehicle):
            return
        try:
            bp = self.bp_lib.find("sensor.other.collision")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(self._on_collision_event)
            self.sensor_list.append(sensor)
        except Exception as e:
            if self.cfg.debug_mode:
                print(f"[WARN] collision sensor setup failed: {e}")

    def _setup_lane_invasion(self) -> None:
        if not self.cfg.enable_lane_invasion_sensor or not self._actor_is_alive(self.vehicle):
            return
        try:
            bp = self.bp_lib.find("sensor.other.lane_invasion")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(self._on_lane_invasion_event)
            self.sensor_list.append(sensor)
        except Exception as e:
            if self.cfg.debug_mode:
                print(f"[WARN] lane invasion sensor setup failed: {e}")

    def _tick(self) -> None:
        self._safe_tick(label="step")
        self._update_spectator()

    def _ego_spawn_candidate_indices(self) -> List[int]:
        n = len(self.spawn_points)
        if n == 0:
            return [0]
        # fixed_spawn_index < 0  → fully random spawn each episode (prevents
        # overfitting to a single scenario; confirmed necessary from training log
        # where 97% collisions all happened at the same map location).
        if self.fixed_spawn_index < 0:
            indices = list(range(n))
            random.shuffle(indices)
            return indices
        # fixed_spawn_index >= 0 → deterministic base + nearby fallbacks
        offsets = [0, 1, 2, 3, 5, 8, 13]
        out: List[int] = []
        seen: set = set()
        for off in offsets:
            idx = (self.fixed_spawn_index + off) % n
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out

    def _route_exists_between(self, start_tf: carla.Transform, goal_index: int) -> bool:
        grp = self._get_global_planner()
        if grp is None:
            return False
        if goal_index < 0 or goal_index >= len(self.spawn_points):
            return False
        try:
            start_wp = self._project_drive_wp(start_tf.location)
            goal_wp = self._project_drive_wp(self.spawn_points[goal_index].location)
            if start_wp is None or goal_wp is None:
                return False
            start_loc = start_wp.transform.location
            goal_loc = goal_wp.transform.location
            if distance2d(start_loc, goal_loc) < self.cfg.candidate_goal_min_dist_m:
                return False
            trace = grp.trace_route(start_loc, goal_loc)
            if len(trace) < 2:
                return False
            route = [wp for wp, _ in trace if wp is not None]
            route = self._dedupe_waypoints(route, min_sep=0.75)
            if len(route) < 2:
                return False
            route_len = self._route_length_from_wps(route)
            return route_len >= self.cfg.min_route_length_m
        except BaseException:
            return False

    def _spawn_ego(self, preferred_goal_index: Optional[int] = None) -> carla.Transform:
        bp = self.bp_lib.find(self.cfg.car_name)
        if bp.has_attribute("color"):
            bp.set_attribute("color", self.cfg.fixed_color)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")

        base_candidates = self._ego_spawn_candidate_indices()
        ordered_indices: List[int] = []
        seen: set[int] = set()

        def _append_idx(idx: int) -> None:
            if idx not in seen:
                ordered_indices.append(idx)
                seen.add(idx)

        for idx in base_candidates:
            _append_idx(idx)
        for idx in range(len(self.spawn_points)):
            _append_idx(idx)

        last_spawn_error: Optional[Exception] = None
        best_route_issue: Optional[str] = None

        for idx in ordered_indices:
            raw_tf = self.spawn_points[idx]
            snapped_tf = self._snap_spawn_to_driving_lane(raw_tf)
            spawn_bases: List[carla.Transform] = [snapped_tf]

            raw_offset = distance2d(raw_tf.location, snapped_tf.location)
            if raw_offset > 0.75:
                spawn_bases.append(raw_tf)
                if self.cfg.debug_mode:
                    print(
                        f"[SPAWN] spawn {idx} snapped to lane center by {raw_offset:.2f}m "
                        f"for map {self.town_name}"
                    )

            if self.cfg.destroy_stale_spawn_blockers:
                cleared = 0
                for base_tf in spawn_bases:
                    cleared += self._clear_spawn_blockers(base_tf)
                if cleared > 0 and self.cfg.debug_mode:
                    print(f"[SPAWN] cleared {cleared} blocking actor(s) near spawn {idx}")

            # Cheap pre-check: skip fixed-goal spawns that definitely cannot route.
            if preferred_goal_index is not None and int(preferred_goal_index) >= 0 and self.cfg.strict_goal_route:
                try:
                    start_wp_hint = self._project_drive_wp(snapped_tf.location)
                    if start_wp_hint is None or (not self._route_exists_between(snapped_tf, int(preferred_goal_index))):
                        best_route_issue = f"No valid fixed-goal route from spawn {idx} to goal {int(preferred_goal_index)}"
                        continue
                except Exception as e:
                    best_route_issue = f"Route precheck failed at spawn {idx}: {e}"
                    continue

            for base_tf in spawn_bases:
                for tf in self._spawn_transform_variants(base_tf):
                    vehicle = None

                    try:
                        vehicle = self.world.try_spawn_actor(bp, tf)
                    except Exception as e:
                        last_spawn_error = e
                        vehicle = None

                    if vehicle is None and self.cfg.destroy_stale_spawn_blockers:
                        cleared = self._clear_spawn_blockers(tf, radius=max(self.cfg.spawn_blocker_radius_m, 3.0))
                        if cleared > 0 and self.cfg.debug_mode:
                            print(f"[SPAWN] retry after clearing {cleared} actor(s) near spawn {idx}")
                        try:
                            vehicle = self.world.try_spawn_actor(bp, tf)
                        except Exception as e:
                            last_spawn_error = e
                            vehicle = None

                    if vehicle is None:
                        continue

                    self.vehicle = vehicle
                    try:
                        self.vehicle.set_autopilot(False)
                    except BaseException:
                        pass
                    self.current_spawn_index = idx
                    self._snap_spectator_to_ego()
                    return self.vehicle.get_transform()

            time.sleep(0.05)

        if preferred_goal_index is not None and int(preferred_goal_index) >= 0 and self.cfg.strict_goal_route and best_route_issue is not None:
            raise RuntimeError(
                f"Failed to find a route-valid spawn for goal {int(preferred_goal_index)} in map {self.town_name}. "
                f"Last issue: {best_route_issue}."
            )
        if last_spawn_error is not None:
            raise RuntimeError(f"Failed to spawn ego vehicle after trying multiple spawn points: {last_spawn_error}")
        raise RuntimeError(
            f"Failed to spawn ego vehicle after trying multiple spawn points in map {self.town_name}. "
            "Spawn locations appear occupied or blocked by residual actors."
        )


    def _spawn_npcs(self, npc_count: int) -> None:
        npc_bps = list(self.bp_lib.filter("vehicle.*"))
        if not npc_bps or self.vehicle is None or npc_count <= 0:
            return
        ego_loc = self.vehicle.get_location()
        goal_loc = self.goal_transform.location if self.goal_transform is not None else None
        candidates = []
        for sp in self.spawn_points:
            if distance2d(sp.location, ego_loc) <= 25.0:
                continue
            if goal_loc is not None and distance2d(sp.location, goal_loc) <= 20.0:
                continue
            candidates.append(sp)
        random.shuffle(candidates)
        for sp in candidates[: max(0, npc_count * 3)]:
            if len(self.npcs) >= npc_count:
                break
            npc_bp = random.choice(npc_bps)
            if npc_bp.has_attribute("role_name"):
                npc_bp.set_attribute("role_name", "autopilot")
            npc = self.world.try_spawn_actor(npc_bp, sp)
            if npc is None:
                continue
            try:
                npc.set_autopilot(True, self.tm.get_port())
                self.tm.vehicle_percentage_speed_difference(npc, random.uniform(-25.0, 10.0))
                self.tm.distance_to_leading_vehicle(npc, random.uniform(2.0, 6.0))
                self.tm.auto_lane_change(npc, random.choice([True, False]))
                self.npcs.append(npc)
            except BaseException:
                self._destroy_actor(npc)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._episode_live = False
        self._terminal_reason = ""

        self._cleanup_episode_actors()
        if self.cfg.destroy_stale_owned_actors_on_reset:
            cleared = self._destroy_residual_owned_actors()
            if cleared > 0 and self.cfg.debug_mode:
                print(f"[RESET] destroyed {cleared} residual owned actor(s) before spawning")
        apply_weather(self.world, self.weather_mode)

        opt = options or {}
        npc_requested = int(opt.get("npc_count", random.randint(self.cfg.npc_min, self.cfg.npc_max)))
        npc_cap = max(int(self.cfg.npc_max), int(self.cfg.train_npc_max))
        npc_count = int(np.clip(npc_requested, 0, npc_cap))
        goal_index_raw = opt.get("goal_index", self.fixed_goal_index if self.cfg.use_fixed_destination else None)
        goal_index = None if goal_index_raw is None or int(goal_index_raw) < 0 else int(goal_index_raw)

        self.episode_steps = 0
        self.prev_action[:] = 0.0
        self.prev_steer = 0.0
        self.prev_acc[:] = 0.0
        self.prev_loc = None
        self.distance_driven_m = 0.0
        self.stuck_steps = 0
        self.release_steps_left = 0
        self.offroute_steps = 0
        self.prev_route_s = None
        self.safety_interventions = 0
        self.blocked_steps_credit = 0
        self._free_stuck_steps = 0
        self.route_progress_idx = 0
        self.route_total_len_m = 0.0

        requested_goal_index = goal_index
        spawn_tf = self._spawn_ego(preferred_goal_index=requested_goal_index)
        start_wp = self.map.get_waypoint(spawn_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if start_wp is None:
            raise RuntimeError("Could not obtain a driving waypoint for the ego spawn location.")

        self._build_route(spawn_tf, start_wp, goal_index=requested_goal_index)

        self._snap_spectator_to_ego()
        for _ in range(max(0, self.cfg.post_spawn_settle_ticks)):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            self._safe_tick(label="reset_settle")
            self._snap_spectator_to_ego()

        for _ in range(max(6, self.cfg.warmup_reset_ticks)):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
            self._tick()

                # If we are using local fallback routing (no planner goal selected),
        # rebuild the local route after the ego has settled on the road.
        if requested_goal_index is None and self.current_goal_index is None:
            s_chk, d_chk, _ = self._route_projection_global(self.vehicle.get_location())
            if d_chk > max(self.cfg.max_reset_start_dL_m, 5.0):
                if self.cfg.debug_mode:
                    pct_chk = 100.0 * s_chk / max(self.route_total_len_m, 1e-6)
                    print(
                        f"[WARN] local fallback route misaligned after settle: "
                        f"progress={pct_chk:.2f}% dL={d_chk:.2f}m; rebuilding from settled ego pose"
                    )
                rebuilt = self._rebuild_local_route_from_current_pose()
                if not rebuilt and self.cfg.debug_mode:
                    print("[WARN] local fallback route rebuild failed; keeping original fallback route")

        s0_global, d0_global, idx0_global = self._route_projection_global(self.vehicle.get_location())
        progress0_pct = 100.0 * s0_global / max(self.route_total_len_m, 1e-6)

        start_to_route0 = d0_global
        if self.route_wps:
            try:
                start_to_route0 = distance2d(self.vehicle.get_location(), self.route_wps[0].transform.location)
            except BaseException:
                pass

        if progress0_pct > self.cfg.max_reset_start_progress_pct or d0_global > self.cfg.max_reset_start_dL_m:
            # For local fallback routes, do not hard-fail reset after rebuild.
            if requested_goal_index is None and self.current_goal_index is None:
                if self.cfg.debug_mode:
                    print(
                        f"[WARN] relaxing reset alignment for local fallback route: "
                        f"progress={progress0_pct:.2f}% dL={d0_global:.2f}m"
                    )
                idx0_global = 0
                s0_global = 0.0
                progress0_pct = 0.0
                d0_global = 0.0
            elif start_to_route0 <= max(self.cfg.max_reset_start_dL_m, 4.0):
                if self.cfg.debug_mode:
                    print(
                        f"[WARN] reset alignment relaxed: progress={progress0_pct:.2f}% dL={d0_global:.2f}m; "
                        "clamping start progress to route origin"
                    )
                idx0_global = 0
                s0_global = 0.0
                progress0_pct = 0.0
            else:
                raise RuntimeError(
                    f"Invalid reset alignment: progress={progress0_pct:.2f}% dL={d0_global:.2f}m "
                    f"route_len={self.route_total_len_m:.2f}m goal_idx={self.current_goal_index}"
                )


        self.route_progress_idx = idx0_global
        self.prev_route_s = s0_global

        self._spawn_npcs(npc_count)

        self._setup_collision()
        self._setup_lane_invasion()
        self._safe_tick(label="post_sensor_attach", raise_on_fail=False)
        self._snap_spectator_to_ego()
        self.prev_loc = self.vehicle.get_location()
        self._teardown_in_progress = False
        self._episode_live = True

        obs = self._get_obs()
        info = {
            "town": self.town_name,
            "npc_count": len(self.npcs),
            "weather": self.weather_mode,
            "spawn_index": self.current_spawn_index,
            "goal_index": self.current_goal_index,
            "requested_goal_index": requested_goal_index,
            "route_total_len_m": float(self.route_total_len_m),
            "carla_server_version": getattr(self.client, "get_server_version", lambda: "unknown")(),
            "carla_client_version": getattr(self.client, "get_client_version", lambda: "unknown")(),
        }
        if self.cfg.debug_mode:
            tf = self.vehicle.get_transform()
            print(
                f"[RESET] town={info['town']} npc={info['npc_count']} weather={info['weather']} "
                f"spawn_idx={info['spawn_index']} goal_idx={info['goal_index']} route_len={info['route_total_len_m']:.1f}m "
                f"spawn=({tf.location.x:.1f},{tf.location.y:.1f},{tf.location.z:.1f}) yaw={tf.rotation.yaw:.1f} "
                f"start_progress={progress0_pct:.2f}% start_dL={d0_global:.2f}"
            )
        return obs, info

    def _compute_reward_components(self, applied_action: np.ndarray) -> Tuple[float, float, float, Dict[str, float]]:
        """Dense multi-objective reward shaping (paper Sec. 3.4, Eqs. 9–13).

        r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u    (Eq. 9)

        r_s: safety  – lane adherence + proximity + red-light  (Eq. 10)
        r_p: progress – tanh(Δs / τ_s) blended with tanh(v̂_t / τ_v)  (Eq. 11)
        r_c: comfort  – quadratic jerk + steer-rate penalty             (Eq. 12)
        r_u: uncertainty – 1 − σ̄  (computed externally, Eq. 13)

        No terminal bonus / penalty (goal_bonus = collision_penalty = 0).
        """
        assert self.vehicle is not None
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed = vec3_length(vel)
        wp_onlane = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_curv = self._curvature_ahead(wp_onlane)
        lane_curv_s = float(np.tanh(10.0 * lane_curv))
        lane_head = self._lane_heading_offset(wp_onlane)
        s_arc, dL = self._route_projection(loc)
        route_obs = self._route_observation_features()
        route_heading_err = float(route_obs.get('route_heading', 0.0)) * (math.pi / 2.0)
        route_cte_norm = float(route_obs.get('route_cte', 0.0))
        delta_s = 0.0 if self.prev_route_s is None else max(0.0, s_arc - self.prev_route_s)

        fog_norm = get_fog_norm(self.world)
        density_norm = float(np.clip(len(self.npcs) / max(self.cfg.npc_max, 1), 0.0, 1.0))
        curv_norm = float(np.clip(abs(lane_curv_s), 0.0, 1.0))
        # μ_A: corridor membership (Eq. 3) – tighter in crowded / foggy / curved contexts
        muA = float(np.clip(1.0 - (0.45 * density_norm + 0.35 * fog_norm + 0.20 * curv_norm), 0.0, 1.0))
        eps_mu = self.cfg.eps_min + (self.cfg.eps_max - self.cfg.eps_min) * muA
        dL_eff = dL / max(eps_mu, 1e-3)
        # ψ_L: soft lane-adherence barrier (Eq. 10)
        psiL = math.tanh(dL_eff / max(self.cfg.tau_d, 1e-3))

        # Proximity and TTC (Eq. 4 / Eq. 10)
        nearest = self.cfg.entity_max_dist
        tau_t = 999.0
        ego_vel = np.array([vel.x, vel.y], dtype=np.float32)
        for npc in self.npcs:
            if npc is None or not npc.is_alive:
                continue
            nloc = npc.get_location()
            dist = distance2d(loc, nloc)
            if dist < nearest:
                nearest = dist
            nvel = npc.get_velocity()
            rel_p = np.array([nloc.x - loc.x, nloc.y - loc.y], dtype=np.float32)
            rel_v = np.array([nvel.x, nvel.y], dtype=np.float32) - ego_vel
            closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
            if dist > 0.5:
                tau = dist / max(closing, 1e-3) if closing > 1e-6 else 999.0
                tau_t = min(tau_t, tau)
        for walker in self._nearby_walkers(loc):
            try:
                wloc = walker.get_location()
                dist = distance2d(loc, wloc)
                nearest = min(nearest, dist)
                wvel = walker.get_velocity()
                rel_p = np.array([wloc.x - loc.x, wloc.y - loc.y], dtype=np.float32)
                rel_v = np.array([wvel.x, wvel.y], dtype=np.float32) - ego_vel
                closing = max(0.0, -float(np.dot(rel_p, rel_v)) / max(float(np.linalg.norm(rel_p)), 1e-6))
                if dist > 0.5:
                    tau = dist / max(closing, 1e-3) if closing > 1e-6 else 999.0
                    tau_t = min(tau_t, tau)
            except BaseException:
                continue
        # ψ_P: proximity surrogate ψ_P = exp(-dist/τ_p)  (Eq. 10)
        psiP = math.exp(-nearest / max(self.cfg.tau_p, 1e-3))

        # Red-light noncompliance ρ_t  (Eq. 10)
        rho = 0.0
        tl_state, tl_dist = self._get_active_traffic_light_info()
        if tl_state == "Red" and tl_dist < self.cfg.tl_stop_dist:
            z1 = float(np.clip((self.cfg.tl_stop_dist - tl_dist) / self.cfg.tl_stop_dist, 0.0, 1.0))
            z2 = float(np.clip(speed / 3.0, 0.0, 1.0))
            rho = float(np.clip(z1 * z2, 0.0, 1.0))

        # Eq. 10: r_s = 1 − κ_L ψ_L − κ_P ψ_P − κ_R ρ_t
        rs = float(np.clip(1.0 - self.cfg.k_l * psiL - self.cfg.k_p * psiP - self.cfg.k_r * rho, -2.0, 1.0))

        # Eq. 11: r_p = tanh(Δs / τ_s)  OR  tanh(v̂_t / τ_v)
        # Use a blend: arc-length progress is primary; velocity projection adds
        # a live signal that rewards maintaining forward speed even between route
        # waypoints, which is important for consistent 200 m completion.
        rp_arc = float(math.tanh(delta_s / max(self.cfg.tau_s, 1e-3)))
        # Velocity projection onto route tangent v̂_t (Eq. 11 alternative)
        v_hat = float(route_obs.get('v_route', 0.0)) * 10.0  # de-normalise
        rp_vel = float(math.tanh(v_hat / max(self.cfg.tau_v, 1e-3)))
        # Blend: 70 % arc-length (reliable), 30 % velocity (dense signal)
        rp = float(0.70 * rp_arc + 0.30 * rp_vel)

        # Eq. 12: r_c = −κ_j j_t² − κ_δ δ̇_t²
        acc = self.vehicle.get_acceleration()
        acc2 = np.array([acc.x, acc.y], dtype=np.float32)
        jerk = float(np.linalg.norm(acc2 - self.prev_acc) / max(self.cfg.dt, 1e-3))
        steer_rate = float((applied_action[2] - self.prev_steer) / max(self.cfg.dt, 1e-3))
        lane_cross = 1.0 if len(self.lane_invasion_events) > 0 else 0.0

        rc_raw = -(self.cfg.k_j * (jerk ** 2)) - (self.cfg.k_delta * (steer_rate ** 2))
        rc = float(np.clip(rc_raw, -5.0, 0.0))

        return rs, rp, rc, {
            "dL": float(dL),
            "route_s": float(s_arc),
            "delta_s": float(delta_s),
            "muA": float(muA),
            "eps_mu": float(eps_mu),
            "psiL": float(psiL),
            "nearest_entity_dist": float(nearest),
            "psiP": float(psiP),
            "rho_red": float(rho),
            "jerk": float(jerk),
            "steer_rate": float(steer_rate),
            "lane_curv": float(lane_curv_s),
            "lane_head": float(lane_head),
            "route_heading_err": float(route_heading_err),
            "route_cte_norm": float(route_cte_norm),
            "lane_cross": float(lane_cross),
            "rc_raw": float(rc_raw),
            "rp_arc": float(rp_arc),
            "rp_vel": float(rp_vel),
            "time_to_conflict": float(tau_t),
            "tl_state": tl_state,
            "tl_dist": float(tl_dist) if tl_dist < 1e8 else -1.0,
        }

    def step(self, action: np.ndarray):
        if self._teardown_in_progress:
            raise RuntimeError("step() called while teardown is in progress")
        if not self._actor_is_alive(self.vehicle):
            raise RuntimeError("step() called with missing/destroyed ego actor")

        self.episode_steps += 1
        self.lane_invasion_events.clear()

        raw = np.array(action, dtype=np.float32).copy()
        raw[0] = float(np.clip(raw[0], 0.0, 1.0))
        raw[1] = float(np.clip(raw[1], 0.0, 1.0))
        raw[2] = float(np.clip(raw[2], -1.0, 1.0))
        if raw[0] < self.cfg.min_throttle_deadzone:
            raw[0] = 0.0
        if raw[1] < self.cfg.min_brake_deadzone:
            raw[1] = 0.0
        if raw[0] > 0.0 and raw[1] > 0.0:
            if raw[0] >= raw[1]:
                raw[1] = 0.0
            else:
                raw[0] = 0.0

        if self.cfg.use_safety_shield:
            applied, route_meta, shield_active = self._safety_filter(raw)
        else:
            applied, route_meta, shield_active = self._policy_passthrough_filter(raw)
        applied[0] = float(np.clip(applied[0], 0.0, 1.0))
        applied[1] = float(np.clip(applied[1], 0.0, 1.0))
        applied[2] = float(np.clip(applied[2], -1.0, 1.0))
        if applied[0] > 0.0 and applied[1] > 0.0:
            if applied[0] >= applied[1]:
                applied[1] = 0.0
            else:
                applied[0] = 0.0

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=float(applied[0]),
                steer=float(applied[2]),
                brake=float(applied[1]),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )
        self._tick()

        cur_loc = self.vehicle.get_location()
        if self.prev_loc is not None:
            self.distance_driven_m += distance2d(cur_loc, self.prev_loc)
        self.prev_loc = cur_loc

        rs, rp, rc, comp = self._compute_reward_components(applied)
        terminated = False
        truncated = False
        reason: Optional[str] = None
        if self.cfg.terminate_on_collision and len(self.collision_events) > 0:
            terminated = True
            reason = "collision"

        dL_after = float(comp.get("dL", 0.0))
        eps_mu_now = float(comp.get("eps_mu", self.cfg.eps_max))
        route_heading_err_now = abs(float(route_meta.get("route_heading_err", 0.0)))
        offroute_thresh = max(self.cfg.max_route_deviation_m, eps_mu_now + 0.75)
        strong_offroute = dL_after > (self.cfg.strong_offroute_factor * offroute_thresh)
        offroute_bad = strong_offroute or ((dL_after > offroute_thresh) and (route_heading_err_now > self.cfg.offroute_heading_gate_rad))
        if offroute_bad:
            self.offroute_steps += 1
        else:
            self.offroute_steps = max(0, self.offroute_steps - 1)
        if self.cfg.terminate_on_offroad and self.offroute_steps >= self.cfg.offroute_grace_steps:
            terminated = True
            reason = "off_route"

        s_arc_now = float(comp.get("route_s", 0.0))
        remaining_s = max(0.0, self.route_total_len_m - s_arc_now)
        goal_euclid = 1e9
        if self.goal_transform is not None:
            goal_euclid = float(cur_loc.distance(self.goal_transform.location))
        route_completion_pct = self._route_completion_pct(s_arc_now)
        success = False
        route_success = route_completion_pct >= self.cfg.route_success_pct
        near_goal = remaining_s <= self.cfg.goal_reach_remaining_s_m and goal_euclid <= self.cfg.goal_reach_dist_m
        # Also count near-goal + high completion as success even if arc-length
        # percentage is slightly below the threshold (route measurement noise).
        near_goal_completion = near_goal and route_completion_pct >= max(88.0, self.cfg.route_success_pct - 7.0)
        if route_success or near_goal_completion:
            terminated = True
            reason = "goal"
            success = True

        blocked_wait = (
            (str(comp.get("tl_state", "None")) == "Red" and (3.6 * vec3_length(self.vehicle.get_velocity())) <= self.cfg.blocked_timeout_speed_kmh)
            or (
                safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0) > 0.0
                and safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0) < self.cfg.front_vehicle_soft_block_dist
                and (3.6 * vec3_length(self.vehicle.get_velocity())) <= self.cfg.blocked_timeout_speed_kmh
            )
        )
        if blocked_wait:
            self.blocked_steps_credit = min(self.blocked_steps_credit + 1, self.cfg.blocked_timeout_extension_steps)
        else:
            self.blocked_steps_credit = max(0, self.blocked_steps_credit - 1)

        tl_now = str(comp.get("tl_state", "None"))
        front_dist_now = safe_float(route_meta.get("safety_front_vehicle_dist", -1.0), -1.0)
        genuinely_open = (
            tl_now not in ("Red", "Yellow")
            and (front_dist_now < 0.0 or front_dist_now > self.cfg.front_vehicle_soft_block_dist)
        )
        if (3.6 * vec3_length(self.vehicle.get_velocity())) < self.cfg.stuck_speed_kmh and genuinely_open:
            self._free_stuck_steps = getattr(self, "_free_stuck_steps", 0) + 1
        else:
            self._free_stuck_steps = 0

        if (not terminated) and self._free_stuck_steps >= self.cfg.stuck_terminate_steps:
            truncated = True
            reason = "stuck"
            self._free_stuck_steps = 0

        near_goal_extra = self.cfg.near_goal_timeout_extension_steps if remaining_s < self.cfg.near_goal_remaining_s_m else 0
        hard_timeout_steps = self.cfg.max_episode_steps + self.blocked_steps_credit + near_goal_extra
        if (not terminated) and (self.episode_steps >= hard_timeout_steps):
            truncated = True
            reason = "timeout"

        self.prev_route_s = s_arc_now
        if shield_active:
            self.safety_interventions += 1
        self.prev_action = applied.copy()
        self.prev_steer = float(applied[2])
        acc = self.vehicle.get_acceleration()
        self.prev_acc = np.array([acc.x, acc.y], dtype=np.float32)

        obs = self._get_obs()
        info = {
            "rs": float(rs),
            "rp": float(rp),
            "rc": float(rc),
            "term_reason": reason or "",
            "goal_reached": bool(reason == "goal"),
            "collision": bool(reason == "collision"),
            "off_road": bool(reason == "off_route"),
            "off_route": bool(reason == "off_route"),
            "timeout": bool(reason in ("timeout", "stuck")),
            "stuck": bool(reason == "stuck"),
            "success": bool(success),
            "steps": int(self.episode_steps),
            "goal_dist": float(remaining_s),
            "goal_euclid": float(goal_euclid),
            "route_completion_pct": float(route_completion_pct),
            "route_total_len_m": float(self.route_total_len_m),
            "stuck_steps": int(self.stuck_steps),
            "offroute_steps": int(self.offroute_steps),
            "blocked_steps_credit": int(self.blocked_steps_credit),
            "shield_active": int(bool(shield_active)),
            "safety_intervention": int(bool(shield_active)),
            "intervention_rate": float(self.safety_interventions / max(self.episode_steps, 1)),
            "desired_speed_kmh": float(route_meta.get("desired_speed_kmh", 0.0)),
            "route_idx": float(route_meta.get("route_idx", 0.0)),
            "route_target_idx": float(route_meta.get("route_target_idx", 0.0)),
            "route_cte": float(route_meta.get("route_cte", 0.0)),
            "route_heading_err": float(route_meta.get("route_heading_err", 0.0)),
            "route_remaining_s": float(route_meta.get("route_remaining_s", remaining_s)),
            "route_policy_blend": float(route_meta.get("policy_route_blend", 0.0)),
            "spawn_index": self.current_spawn_index,
            "goal_index": self.current_goal_index,
            "applied_action": applied.copy(),
            "distance_driven_m": float(self.distance_driven_m),
        }
        info.update(comp)
        # ------------------------------------------------------------------
        # σ̄ proxy for env-time reward (Eq. 13: r_u = 1 − σ̄).
        # During rollout the critic ensemble is not called every step for
        # efficiency; instead we compute a lightweight proxy.
        # Proxy: σ̄ ≈ 1 − μ_A (fragile contexts → higher uncertainty).
        # The training loop overwrites this with the real ensemble-based σ̄
        # via recompute_agent_reward() so the stored replay reward is correct.
        # ------------------------------------------------------------------
        muA_val = float(np.clip(safe_float(info.get("muA", 0.5)), 0.0, 1.0))
        # Also incorporate nearest-entity distance for a richer proxy:
        # closer entities → higher uncertainty even in clear weather.
        near_norm = float(np.clip(safe_float(info.get("nearest_entity_dist", self.cfg.entity_max_dist))
                                  / max(self.cfg.entity_max_dist, 1.0), 0.0, 1.0))
        # sigma_proxy ↑ when μ_A ↓ (fog/dense) or entities are very close
        sigma_proxy = float(np.clip((1.0 - muA_val) * 0.70 + (1.0 - near_norm) * 0.30, 0.0, 1.0))
        reward = build_reward_from_info(info, sigma_bar=sigma_proxy, cfg=self.cfg)
        info["sigma_proxy"] = float(sigma_proxy)
        info["env_reward_proxy"] = float(reward)

        if self.cfg.debug_mode and (self.episode_steps == 1 or self.episode_steps % self.cfg.debug_step_freq == 0 or terminated or truncated):
            vel = self.vehicle.get_velocity()
            speed_kmh = 3.6 * vec3_length(vel)
            print(
                f"[STEP] step={self.episode_steps:04d} speed={speed_kmh:6.2f}km/h "
                f"thr={applied[0]:.2f} brk={applied[1]:.2f} str={applied[2]:+.2f} safe={int(bool(shield_active))} "
                f"progress={route_completion_pct:.1f}% goal={remaining_s:.2f}m dL={dL_after:.2f} "
                f"ttc={safe_float(info.get('time_to_conflict', 999.0)):.2f}s tl={info.get('tl_state', 'None')} "
                f"credit={int(self.blocked_steps_credit)} reason={reason or ''}"
            )
        if terminated or truncated:
            self._begin_episode_teardown(reason or ("timeout" if truncated else "terminated"))    
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self._cleanup_episode_actors()
        except BaseException:
            pass
        try:
            self._restore_world_settings()
        except BaseException:
            pass


# ======================================================
# Networks / Agent
# ======================================================
class GraphAttention(nn.Module):
    """Uncertainty-weighted causal attention over ego-node edges.

    Paper Eq. 7:
        α_i = softmax_i( -‖Δp_i‖² / (σ_i² + ε) )
        z_t  = Σ_i α_i W_e e_i^t

    The attention weight penalises distant AND uncertain actors:
    higher σ_i² (noisier observation) → smaller denominator amplification
    → the actor receives lower attention weight.
    """

    def __init__(self, edge_dim: int, z_dim: int = 64):
        super().__init__()
        self.we = nn.Linear(edge_dim, z_dim)
        self.sigma_head = nn.Sequential(nn.Linear(edge_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, edges: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Per-edge aleatoric variance σ_i² (Eq. 6: last feature is σ²_i)
        base_sigma2 = edges[..., -1].clamp(min=1e-3)
        sigma_add = F.softplus(self.sigma_head(edges).squeeze(-1))
        sigma2 = base_sigma2 + sigma_add + 1e-3

        # Squared ego-frame displacement ‖Δp_i‖²
        dp = edges[..., 0:2]
        dp2 = (dp * dp).sum(dim=-1)

        # Paper Eq. 7: logit = -‖Δp_i‖² / (σ_i² + ε)
        # (original had an extra factor of 2 in the denominator; removed for paper fidelity)
        logits = -dp2 / (sigma2 + 1e-6)

        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(mask > 0.5, logits, torch.full_like(logits, neg_inf))
        all_masked = mask.sum(dim=1, keepdim=True) < 0.5
        if all_masked.any():
            logits = logits.clone()
            logits[all_masked.squeeze(1), 0] = 0.0

        # α_i = softmax(logits)  →  z_t = Σ_i α_i W_e e_i^t
        alpha = torch.softmax(logits, dim=1)
        eproj = self.we(edges)
        z = torch.sum(alpha.unsqueeze(-1) * eproj, dim=1)

        # Aggregated aleatoric variance σ²_ale (Eq. 8, injected into state s_t)
        denom = mask.sum(dim=1).clamp(min=1.0)
        sigma_ale = torch.sum(sigma2 * mask, dim=1) / denom
        return z, alpha, sigma_ale


class CompactStateEncoder(nn.Module):
    """Ego-centric relational state encoder (paper Sec. 3.3, Eqs. 6–8).

    s_t = [z_t; v_ego; a^{t-1}_ego; d_goal; φ_lane; σ²_ale]   (Eq. 8)

    z_t is the uncertainty-weighted interaction embedding (Eq. 7).
    σ²_ale is appended so the actor and critic see raw aleatoric variance.
    """

    def __init__(self, cfg: Config = CFG, z_dim: int = 64, scalar_dim: Optional[int] = None, out_dim: int = 256):
        super().__init__()
        scalar_dim = int(cfg.scalar_dim if scalar_dim is None else scalar_dim)
        self.graph = GraphAttention(edge_dim=cfg.edge_dim, z_dim=z_dim)
        self.mlp_scal = nn.Sequential(
            nn.Linear(scalar_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(z_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, alpha, sigma_ale = self.graph(edges, mask)
        # Append σ²_ale to scalars (Eq. 8: σ²_ale term in s_t)
        scal_plus = torch.cat([scalars, sigma_ale.unsqueeze(-1)], dim=-1)
        s = self.mlp_scal(scal_plus)
        h = self.fuse(torch.cat([z, s], dim=-1))
        return h, alpha, sigma_ale


class Actor(nn.Module):
    def __init__(self, enc: CompactStateEncoder, action_dim: int = 3):
        super().__init__()
        self.enc = enc
        self.mu = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.logstd = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.register_buffer("action_scale", torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32))

    def forward(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor):
        h, alpha, sigma_ale = self.enc(scalars, edges, mask)
        mu = self.mu(h)
        logstd = self.logstd(h).clamp(-5.0, 1.0)
        return mu, logstd, alpha, sigma_ale

    def _squash_action(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tanh_u = torch.tanh(u)
        action = tanh_u * self.action_scale + self.action_bias
        log_scale = torch.log(self.action_scale.clamp(min=1e-6)).sum()
        log_det_tanh = torch.log((1.0 - tanh_u.pow(2)).clamp(min=1e-6)).sum(dim=-1)
        log_abs_det = log_scale + log_det_tanh
        return action, log_abs_det

    def sample(
        self,
        scalars: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd, alpha, sigma_ale = self.forward(scalars, edges, mask)
        std = logstd.exp()
        normal = torch.distributions.Normal(mu, std)
        u = mu if deterministic else normal.rsample()
        action, log_abs_det = self._squash_action(u)
        log_pi: Optional[torch.Tensor] = None
        if with_logprob:
            log_pi = normal.log_prob(u).sum(dim=-1) - log_abs_det
        return action, log_pi, mu, logstd, alpha, sigma_ale

    def act_deterministic(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        action, _, _, _, _, _ = self.sample(scalars, edges, mask, deterministic=True, with_logprob=False)
        return action


class Critic(nn.Module):
    def __init__(self, enc: CompactStateEncoder):
        super().__init__()
        self.enc = enc
        self.q = nn.Sequential(nn.Linear(256 + 3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h, _, _ = self.enc(scalars, edges, mask)
        return self.q(torch.cat([h, action], dim=-1)).squeeze(-1)

class ReplayBuffer:
    def __init__(self, capacity: int, cfg: Config = CFG):
        self.capacity = int(capacity)
        self.cfg = cfg
        self.ptr = 0
        self.size = 0
        self.scalars = np.zeros((self.capacity, cfg.scalar_dim), dtype=np.float32)
        self.edges = np.zeros((self.capacity, cfg.max_entities, cfg.edge_dim), dtype=np.float32)
        self.mask = np.zeros((self.capacity, cfg.max_entities), dtype=np.float32)
        self.actions = np.zeros((self.capacity, cfg.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_scalars = np.zeros((self.capacity, cfg.scalar_dim), dtype=np.float32)
        self.next_edges = np.zeros((self.capacity, cfg.max_entities, cfg.edge_dim), dtype=np.float32)
        self.next_mask = np.zeros((self.capacity, cfg.max_entities), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.size)

    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray, reward: float, next_obs: Dict[str, np.ndarray], done: bool) -> None:
        i = self.ptr
        self.scalars[i] = obs["scalars"]
        self.edges[i] = obs["edges"]
        self.mask[i] = obs["mask"]
        self.actions[i] = action
        self.rewards[i] = float(reward)
        self.next_scalars[i] = next_obs["scalars"]
        self.next_edges[i] = next_obs["edges"]
        self.next_mask[i] = next_obs["mask"]
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.size <= 0:
            raise ValueError("ReplayBuffer is empty.")
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "scalars": torch.from_numpy(self.scalars[idx]).to(device).float(),
            "edges": torch.from_numpy(self.edges[idx]).to(device).float(),
            "mask": torch.from_numpy(self.mask[idx]).to(device).float(),
            "actions": torch.from_numpy(self.actions[idx]).to(device).float(),
            "rewards": torch.from_numpy(self.rewards[idx]).to(device).float(),
            "next_scalars": torch.from_numpy(self.next_scalars[idx]).to(device).float(),
            "next_edges": torch.from_numpy(self.next_edges[idx]).to(device).float(),
            "next_mask": torch.from_numpy(self.next_mask[idx]).to(device).float(),
            "dones": torch.from_numpy(self.dones[idx]).to(device).float(),
        }


class UncertaintyCalibrator:
    def __init__(self, temperature: float = 1.0):
        self.center = 0.5
        self.scale = 0.25
        self.min_val = 0.0
        self.max_val = 1.0
        self.temperature = float(max(temperature, 1e-3))
        self.fitted = False

    def fit_from_values(self, values: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {
                "fitted": 0.0,
                "center": float(self.center),
                "scale": float(self.scale),
                "min_val": float(self.min_val),
                "max_val": float(self.max_val),
            }
        q_lo = float(np.clip(q_lo, 0.0, 1.0))
        q_hi = float(np.clip(q_hi, q_lo + 1e-6, 1.0))
        lo = float(np.quantile(arr, q_lo))
        hi = float(np.quantile(arr, q_hi))
        if not np.isfinite(lo):
            lo = float(np.min(arr))
        if not np.isfinite(hi):
            hi = float(np.max(arr))
        if hi <= lo:
            hi = lo + 1e-3
        arr_mm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        center = float(np.median(arr_mm))
        mad = float(np.median(np.abs(arr_mm - center)))
        scale = max(1.4826 * mad, 0.05)
        self.center = center
        self.scale = scale
        self.min_val = lo
        self.max_val = hi
        self.fitted = True
        return {
            "fitted": 1.0,
            "center": self.center,
            "scale": self.scale,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.fitted:
            x = (x - self.min_val) / max(self.max_val - self.min_val, 1e-6)
            x = x.clamp(0.0, 1.0)
        z = (x - self.center) / max(self.scale * self.temperature, 1e-6)
        return torch.sigmoid(z)

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not state:
            return
        self.center = float(state.get("center", self.center))
        self.scale = float(max(float(state.get("scale", self.scale)), 1e-6))
        self.min_val = float(state.get("min_val", self.min_val))
        self.max_val = float(max(float(state.get("max_val", self.max_val)), self.min_val + 1e-6))
        self.temperature = float(max(float(state.get("temperature", self.temperature)), 1e-6))
        self.fitted = bool(state.get("fitted", self.fitted))

    def state_dict(self) -> Dict[str, object]:
        return {
            "center": float(self.center),
            "scale": float(self.scale),
            "min_val": float(self.min_val),
            "max_val": float(self.max_val),
            "temperature": float(self.temperature),
            "fitted": bool(self.fitted),
        }


class SACAgent:
    def __init__(self, device: torch.device, cfg: Config = CFG):
        self.device = device
        self.cfg = cfg
        self.actor = Actor(CompactStateEncoder(cfg=cfg).to(device), action_dim=cfg.action_dim).to(device)
        self.critics = nn.ModuleList([Critic(CompactStateEncoder(cfg=cfg).to(device)).to(device) for _ in range(cfg.n_critics)])
        self.target_critics = nn.ModuleList([Critic(CompactStateEncoder(cfg=cfg).to(device)).to(device) for _ in range(cfg.n_critics)])
        for k in range(cfg.n_critics):
            self.target_critics[k].load_state_dict(self.critics[k].state_dict())
        self.sigma_cal = UncertaintyCalibrator(temperature=cfg.calib_temperature)

        critic_params: List[nn.Parameter] = []
        for critic in self.critics:
            critic_params.extend(list(critic.parameters()))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(critic_params, lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(max(cfg.init_alpha, 1e-6)), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = -float(cfg.action_dim)
        self.training_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def train(self) -> None:
        self.actor.train()
        self.critics.train()
        self.target_critics.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critics.eval()
        self.target_critics.eval()

    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        scalars = torch.from_numpy(obs["scalars"][None]).to(self.device).float()
        edges = torch.from_numpy(obs["edges"][None]).to(self.device).float()
        mask = torch.from_numpy(obs["mask"][None]).to(self.device).float()
        if deterministic:
            action = self.actor.act_deterministic(scalars, edges, mask)
        else:
            action, _, _, _, _, _ = self.actor.sample(scalars, edges, mask, deterministic=False, with_logprob=False)
        return action[0].detach().cpu().numpy().astype(np.float32)

    def critic_ensemble(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor, action: torch.Tensor, target: bool = False) -> torch.Tensor:
        critics = self.target_critics if target else self.critics
        qs = [critics[k](scalars, edges, mask, action) for k in range(self.cfg.n_critics)]
        return torch.stack(qs, dim=0)

    @torch.no_grad()
    def critic_ensemble_nograd(self, scalars, edges, mask, action, target: bool = False) -> torch.Tensor:
        return self.critic_ensemble(scalars, edges, mask, action, target=target)

    def compute_sigma_dec(self, sigma_ale: torch.Tensor, qs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_epi = qs.var(dim=0, unbiased=False)
        sigma_dec = sigma_ale + sigma_epi
        return sigma_dec, sigma_epi

    def compute_sigma_bar(self, sigma_ale: torch.Tensor, qs: torch.Tensor, calibrator: Optional[UncertaintyCalibrator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_dec, sigma_epi = self.compute_sigma_dec(sigma_ale, qs)
        cal = self.sigma_cal if calibrator is None else calibrator
        sigma_bar = cal.transform_tensor(sigma_dec)
        return sigma_bar, sigma_epi

    def uncertainty_features(self, sigma_bar: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        edge_var = edges[..., -1]
        weight = mask.float()
        denom = weight.sum(dim=-1).clamp(min=1.0)
        mean_edge = (edge_var * weight).sum(dim=-1) / denom
        centered = (edge_var - mean_edge.unsqueeze(-1)) * weight
        std_edge = torch.sqrt((centered.pow(2).sum(dim=-1) / denom).clamp(min=1e-6))
        return torch.stack([sigma_bar, mean_edge, std_edge], dim=-1)

    @torch.no_grad()
    def fit_uncertainty_calibrator(self, holdout_batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        sigma_values: List[np.ndarray] = []
        prev_training = self.actor.training
        self.eval()
        for batch in holdout_batches:
            action, _, _, _, _, sigma_ale = self.actor.sample(batch["scalars"], batch["edges"], batch["mask"], deterministic=True, with_logprob=False)
            qs = self.critic_ensemble_nograd(batch["scalars"], batch["edges"], batch["mask"], action)
            sigma_dec, _ = self.compute_sigma_dec(sigma_ale, qs)
            sigma_values.append(sigma_dec.reshape(-1).detach().cpu().numpy())
        if prev_training:
            self.train()
        if not sigma_values:
            return {"calib_fitted": 0.0, "calib_center": float(self.sigma_cal.center), "calib_scale": float(self.sigma_cal.scale)}
        stats = self.sigma_cal.fit_from_values(
            np.concatenate(sigma_values, axis=0),
            q_lo=self.cfg.calib_quantile_lo,
            q_hi=self.cfg.calib_quantile_hi,
        )
        return {
            "calib_fitted": float(stats.get("fitted", 0.0)),
            "calib_center": float(stats.get("center", self.sigma_cal.center)),
            "calib_scale": float(stats.get("scale", self.sigma_cal.scale)),
            "calib_min": float(stats.get("min_val", self.sigma_cal.min_val)),
            "calib_max": float(stats.get("max_val", self.sigma_cal.max_val)),
        }

    @torch.no_grad()
    def select_beta0_from_holdout(
        self,
        holdout_batches: Sequence[Dict[str, torch.Tensor]],
        candidates: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        if not holdout_batches:
            return {"beta0_selected": float(self.cfg.beta0), "beta0_validation_loss": 0.0}
        cand_list = [float(x) for x in (self.cfg.beta0_candidates if candidates is None else candidates)]
        prev_beta0 = float(self.cfg.beta0)
        prev_training = self.actor.training
        self.eval()
        best_beta0 = prev_beta0
        best_loss: Optional[float] = None
        for cand in cand_list:
            self.cfg.beta0 = float(cand)
            losses: List[float] = []
            for batch in holdout_batches:
                loss, _ = self.compute_actor_loss(batch["scalars"], batch["edges"], batch["mask"])
                losses.append(float(loss.detach().cpu().item()))
            score = float(np.mean(losses)) if losses else float("inf")
            if best_loss is None or score < best_loss:
                best_loss = score
                best_beta0 = float(cand)
        self.cfg.beta0 = best_beta0
        if prev_training:
            self.train()
        return {"beta0_selected": float(best_beta0), "beta0_validation_loss": float(best_loss if best_loss is not None else 0.0)}

    def _actor_rl_loss_for_actor(self, actor: "Actor", batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        action, log_pi, _, _, _, sigma_ale = actor.sample(batch["scalars"], batch["edges"], batch["mask"], deterministic=False, with_logprob=True)
        qs = self.critic_ensemble_nograd(batch["scalars"], batch["edges"], batch["mask"], action)
        min_q = qs.min(dim=0)[0]
        sigma_bar, _ = self.compute_sigma_bar(sigma_ale, qs.detach())
        beta = self.cfg.beta0 * (1.0 - sigma_bar.detach())
        effective_alpha = self.alpha.detach() + self.cfg.lambda_ent * beta
        return (effective_alpha * log_pi - min_q).mean()

    @staticmethod
    def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
        x = x.float()
        y = y.float()
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        zz = torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz
        bandwidth = 1.0
        XX = torch.zeros_like(xx)
        YY = torch.zeros_like(yy)
        XY = torch.zeros_like(zz)
        for _ in range(kernel_num):
            XX = XX + torch.exp(-dxx / bandwidth)
            YY = YY + torch.exp(-dyy / bandwidth)
            XY = XY + torch.exp(-dxy / bandwidth)
            bandwidth *= kernel_mul
        return (XX.mean() + YY.mean() - 2.0 * XY.mean()).clamp(min=0.0)

    def compute_actor_loss(self, scalars: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Actor loss with uncertainty-gated entropy (paper Eq. 15).

        L_ent = −β(σ̄) H(π_θ(·|s_t))
        β(σ̄)  = β₀(1 − σ̄)          → low confidence ↑ σ̄ → smaller β → less entropy

        Effective temperature: α_eff = α + λ_ent · β(σ̄)
        Combined actor loss  : L_actor = E[α_eff · log π − min_k Q_k]
        """
        action, log_pi, _, _, _, sigma_ale = self.actor.sample(scalars, edges, mask, deterministic=False, with_logprob=True)
        qs = self.critic_ensemble(scalars, edges, mask, action, target=False)
        min_q = qs.min(dim=0)[0]
        sigma_bar, _ = self.compute_sigma_bar(sigma_ale, qs.detach())
        # Eq. 15: β(σ̄) = β₀(1 − σ̄)
        beta = self.cfg.beta0 * (1.0 - sigma_bar.detach())
        effective_alpha = self.alpha.detach() + self.cfg.lambda_ent * beta
        actor_loss = (effective_alpha * log_pi - min_q).mean()
        stats = {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "log_pi": float(log_pi.detach().mean().cpu().item()),
            "sigma_bar": float(sigma_bar.detach().mean().cpu().item()),
            "beta": float(beta.detach().mean().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
        }
        return actor_loss, stats

    def compute_transfer_loss(
        self,
        source_agent: "SACAgent",
        scalars: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        source_scalars: Optional[torch.Tensor] = None,
        source_edges: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Causal-uncertainty transfer loss (paper Eq. 16).

        L_trans = L_KL + λ_α MMD(α_s, α_t) + λ_u ‖u_s − u_t‖²

        L_KL  : KL divergence between source and target action distributions
        MMD   : Maximum Mean Discrepancy on relational attention vectors (α)
        ‖u‖²  : MSE on uncertainty feature vectors (σ̄, mean σ², std σ²)

        Combined with MAML initialisation (Eq. 17) in maml_style_initialize().
        Full objective (Eq. 18): L_RL + λ_ent L_ent + λ_T L_trans
        """
        s_scalars = scalars if source_scalars is None else source_scalars
        s_edges = edges if source_edges is None else source_edges
        s_mask = mask if source_mask is None else source_mask

        mu_t, logstd_t, alpha_t, sigma_ale_t = self.actor.forward(scalars, edges, mask)
        mu_s, logstd_s, alpha_s, sigma_ale_s = source_agent.actor.forward(s_scalars, s_edges, s_mask)
        alpha_t = alpha_t.reshape(alpha_t.shape[0], -1)
        alpha_s = alpha_s.reshape(alpha_s.shape[0], -1)

        dist_t = torch.distributions.Normal(mu_t, logstd_t.exp())
        with torch.no_grad():
            dist_s = torch.distributions.Normal(mu_s, logstd_s.exp())
            action_s, _, _, _, _, _ = source_agent.actor.sample(s_scalars, s_edges, s_mask, deterministic=True, with_logprob=False)
            qs_s = source_agent.critic_ensemble_nograd(s_scalars, s_edges, s_mask, action_s)
            sigma_bar_s, _ = source_agent.compute_sigma_bar(sigma_ale_s, qs_s)
            u_s = source_agent.uncertainty_features(sigma_bar_s, s_edges, s_mask)

        action_t, _, _, _, _, _ = self.actor.sample(scalars, edges, mask, deterministic=True, with_logprob=False)
        qs_t = self.critic_ensemble(scalars, edges, mask, action_t)
        sigma_bar_t, _ = self.compute_sigma_bar(sigma_ale_t, qs_t.detach())
        u_t = self.uncertainty_features(sigma_bar_t, edges, mask)

        kl_loss = torch.distributions.kl.kl_divergence(dist_s, dist_t).sum(dim=-1).mean()
        mmd_loss = self.compute_mmd(alpha_s.detach(), alpha_t)
        u_loss = F.mse_loss(u_t, u_s.detach())
        total = kl_loss + self.cfg.lambda_alpha_align * mmd_loss + self.cfg.lambda_u_align * u_loss
        stats = {
            "transfer_loss": float(total.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "mmd_loss": float(mmd_loss.detach().cpu().item()),
            "u_loss": float(u_loss.detach().cpu().item()),
        }
        return total, stats

    def soft_update_targets(self) -> None:
        tau = float(self.cfg.target_tau)
        with torch.no_grad():
            for critic, target in zip(self.critics, self.target_critics):
                for p, tp in zip(critic.parameters(), target.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        source_agent: Optional["SACAgent"] = None,
        source_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        self.train()
        scalars = batch["scalars"]
        edges = batch["edges"]
        mask = batch["mask"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_scalars = batch["next_scalars"]
        next_edges = batch["next_edges"]
        next_mask = batch["next_mask"]
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_pi, _, _, _, next_sigma_ale = self.actor.sample(next_scalars, next_edges, next_mask, deterministic=False, with_logprob=True)
            next_qs = self.critic_ensemble(next_scalars, next_edges, next_mask, next_action, target=True)
            next_q_min = next_qs.min(dim=0)[0]
            next_sigma_bar, _ = self.compute_sigma_bar(next_sigma_ale, next_qs)
            beta_next = self.cfg.beta0 * (1.0 - next_sigma_bar)
            effective_alpha_next = self.alpha.detach() + self.cfg.lambda_ent * beta_next
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * (next_q_min - effective_alpha_next * next_log_pi)

        current_qs = self.critic_ensemble(scalars, edges, mask, actions, target=False)
        critic_loss = sum(F.mse_loss(current_qs[k], target_q) for k in range(self.cfg.n_critics))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critics.parameters()), 10.0)
        self.critic_opt.step()

        actor_loss, actor_stats = self.compute_actor_loss(scalars, edges, mask)
        total_actor_loss = actor_loss
        transfer_stats: Dict[str, float] = {}
        if source_agent is not None:
            transfer_loss, transfer_stats = self.compute_transfer_loss(
                source_agent,
                scalars,
                edges,
                mask,
                None if source_batch is None else source_batch["scalars"],
                None if source_batch is None else source_batch["edges"],
                None if source_batch is None else source_batch["mask"],
            )
            total_actor_loss = total_actor_loss + self.cfg.lambda_transfer * transfer_loss

        self.actor_opt.zero_grad(set_to_none=True)
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        with torch.no_grad():
            _, log_pi_alpha, _, _, _, _ = self.actor.sample(scalars, edges, mask, deterministic=False, with_logprob=True)
        alpha_loss = -(self.log_alpha * (log_pi_alpha + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        # Clamp log_alpha so alpha never collapses below 0.01.
        # Without this floor, alpha -> 0 by step ~18k (confirmed in training log),
        # which kills exploration and causes the policy to deterministically drive
        # into the same NPC at the same spot every episode.
        with torch.no_grad():
            self.log_alpha.clamp_(min=math.log(0.01))

        self.soft_update_targets()
        self.training_steps += 1

        stats = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
        }
        stats.update(actor_stats)
        stats.update(transfer_stats)
        return stats

    def maml_style_initialize(self, domain_batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """MAML-style initialisation for few-shot adaptation (paper Eq. 17).

        θ* = argmin_θ Σ_{d∈D} L^(d)_RL(θ − α∇_θ L^(d)_RL(θ))

        A fast-learner actor is initialised from the current parameters, takes
        one inner-loop gradient step on the support half of each domain batch,
        then its query loss is computed.  The weighted average of parameter
        deltas is applied back to self.actor with step-size maml_meta_step_size.
        """
        if not domain_batches:
            return {"maml_batches": 0.0, "maml_inner_loss": 0.0, "maml_query_loss": 0.0}
        base_state = {k: v.detach().clone() for k, v in self.actor.state_dict().items()}
        delta_accum = {k: torch.zeros_like(v) for k, v in base_state.items()}
        weight_accum = 0.0
        inner_losses: List[float] = []
        query_losses: List[float] = []
        for batch in domain_batches:
            bs = int(batch["scalars"].shape[0])
            if bs >= 4:
                split = max(1, bs // 2)
                support = {k: v[:split] for k, v in batch.items()}
                query = {k: v[split:] for k, v in batch.items()}
                if int(query["scalars"].shape[0]) == 0:
                    query = support
            else:
                support = batch
                query = batch

            fast_actor = Actor(CompactStateEncoder(cfg=self.cfg).to(self.device), action_dim=self.cfg.action_dim).to(self.device)
            fast_actor.load_state_dict(base_state)
            fast_opt = torch.optim.Adam(fast_actor.parameters(), lr=self.cfg.maml_inner_lr)

            support_loss_val = 0.0
            for _ in range(max(1, int(self.cfg.maml_inner_steps))):
                fast_loss = self._actor_rl_loss_for_actor(fast_actor, support)
                support_loss_val = float(fast_loss.detach().cpu().item())
                fast_opt.zero_grad(set_to_none=True)
                fast_loss.backward()
                torch.nn.utils.clip_grad_norm_(fast_actor.parameters(), 10.0)
                fast_opt.step()

            query_loss = self._actor_rl_loss_for_actor(fast_actor, query)
            query_loss_val = float(query_loss.detach().cpu().item())
            inner_losses.append(support_loss_val)
            query_losses.append(query_loss_val)

            weight = 1.0 / (1.0 + max(query_loss_val, 0.0))
            fast_state = fast_actor.state_dict()
            for key in delta_accum:
                delta_accum[key] = delta_accum[key] + weight * (fast_state[key] - base_state[key])
            weight_accum += weight

        if weight_accum > 0.0:
            with torch.no_grad():
                cur_state = self.actor.state_dict()
                step_size = float(max(self.cfg.maml_meta_step_size, 1e-6))
                for key in cur_state:
                    cur_state[key].copy_(base_state[key] + step_size * (delta_accum[key] / weight_accum))
        return {
            "maml_batches": float(len(domain_batches)),
            "maml_inner_loss": float(np.mean(inner_losses)) if inner_losses else 0.0,
            "maml_query_loss": float(np.mean(query_losses)) if query_losses else 0.0,
        }

    def save(self, path: str) -> None:
        ckpt = {
            "actor": self.actor.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "target_critics": [c.state_dict() for c in self.target_critics],
            "sigma_cal": self.sigma_cal.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "alpha_opt": self.alpha_opt.state_dict(),
            "training_steps": int(self.training_steps),
            "cfg": self.cfg.__dict__.copy(),
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        load_module_state_compat(self.actor, ckpt["actor"])
        for k, sd in enumerate(ckpt.get("critics", [])):
            if k < len(self.critics):
                load_module_state_compat(self.critics[k], sd)
        target_sds = ckpt.get("target_critics", [])
        if target_sds:
            for k, sd in enumerate(target_sds):
                if k < len(self.target_critics):
                    load_module_state_compat(self.target_critics[k], sd)
        else:
            for k in range(self.cfg.n_critics):
                self.target_critics[k].load_state_dict(self.critics[k].state_dict())
        self.sigma_cal.load_state_dict(ckpt.get("sigma_cal", {}))
        if "actor_opt" in ckpt:
            try:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
            except Exception:
                pass
        if "critic_opt" in ckpt:
            try:
                self.critic_opt.load_state_dict(ckpt["critic_opt"])
            except Exception:
                pass
        if "alpha_opt" in ckpt:
            try:
                self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
            except Exception:
                pass
        if "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(float(ckpt["log_alpha"]), device=self.device))
        saved_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        if isinstance(saved_cfg, dict):
            for key in ["beta0", "lambda_ent", "lambda_transfer", "lambda_alpha_align", "lambda_u_align", "calib_temperature"]:
                if key in saved_cfg and hasattr(self.cfg, key):
                    setattr(self.cfg, key, type(getattr(self.cfg, key))(saved_cfg[key]))
        self.training_steps = int(ckpt.get("training_steps", 0))
        self.eval()


# ======================================================
# Reward / Metrics
# ======================================================
def build_reward_from_info(info: Dict[str, object], sigma_bar: float, cfg: Config = CFG) -> float:
    """Paper-faithful dense reward from Eqs. (9)–(13).

    r_t = w_s r_s + w_p r_p + w_c r_c + w_u r_u    (Eq. 9)
    r_u = 1 − σ̄                                     (Eq. 13)

    No terminal bonus/penalty (goal_bonus = collision_penalty = 0):
    the Lagrangian shaping in Eq. 5 is handled through continuous smooth
    surrogates in r_s and r_p only.
    """
    rs = safe_float(info.get("rs", 0.0))
    rp = safe_float(info.get("rp", 0.0))
    rc = safe_float(info.get("rc", 0.0))
    ru = 1.0 - float(np.clip(sigma_bar, 0.0, 1.0))   # Eq. 13
    return float(cfg.w_s * rs + cfg.w_p * rp + cfg.w_c * rc + cfg.w_u * ru)


@torch.no_grad()
def infer_sigma_bar(agent: SACAgent, obs: Dict[str, np.ndarray], action: np.ndarray) -> float:
    scalars = torch.from_numpy(obs["scalars"][None]).to(agent.device).float()
    edges = torch.from_numpy(obs["edges"][None]).to(agent.device).float()
    mask = torch.from_numpy(obs["mask"][None]).to(agent.device).float()
    act = torch.from_numpy(action[None]).to(agent.device).float()
    _, _, _, sigma_ale = agent.actor.forward(scalars, edges, mask)
    qs = agent.critic_ensemble_nograd(scalars, edges, mask, act)
    sigma_bar, _ = agent.compute_sigma_bar(sigma_ale.reshape(1), qs.reshape(agent.cfg.n_critics, 1))
    return float(sigma_bar.reshape(-1)[0].cpu().item())


def recompute_agent_reward(agent: SACAgent, obs: Dict[str, np.ndarray], action: np.ndarray, info: Dict[str, object], cfg: Config = CFG) -> Tuple[float, float]:
    sigma_bar = infer_sigma_bar(agent, obs, action)
    reward = build_reward_from_info(info, sigma_bar, cfg=cfg)
    return reward, sigma_bar


def compute_infraction_scores(coll_per_km: float, off_per_km: float, to_per_km: float) -> Tuple[float, float]:
    kappas = [0.02, 0.02, 0.02]
    qkms = [coll_per_km, off_per_km, to_per_km]
    penalties = [max(0.0, 1.0 - min(1.0, q / k)) for q, k in zip(qkms, kappas)]
    iscore = float(np.prod(penalties))
    return iscore, 100.0 * iscore


# ======================================================
# Evaluation helpers
# ======================================================
def build_env_from_args(args: argparse.Namespace, cfg: Config) -> CarlaReliableTransferEnv:
    return CarlaReliableTransferEnv(
        host=args.host,
        port=args.port,
        town_name=args.target_town,
        fixed_spawn_index=args.spawn_index,
        fixed_goal_index=cfg.target_goal_index,
        weather_mode=cfg.target_weather,
        cfg=cfg,
    )


def robust_reset(
    env: CarlaReliableTransferEnv,
    env_builder: Callable[[], CarlaReliableTransferEnv],
    npc_count: int,
    goal_index: int,
    max_tries: int = 5,
) -> Tuple[CarlaReliableTransferEnv, Dict[str, np.ndarray], Dict[str, object]]:
    last_err: Optional[Exception] = None
    cur_env = env
    sleep_secs = [2.0, 4.0, 8.0, 12.0, 20.0]
    for attempt in range(1, max_tries + 1):
        try:
            obs, info = cur_env.reset(options={"npc_count": npc_count, "goal_index": goal_index})
            return cur_env, obs, info
        except BaseException as e:
            last_err = Exception(str(e))
            print(f"[WARN] reset attempt {attempt}/{max_tries} failed: {e}")
        try:
            cur_env.close()
        except BaseException:
            pass
        if attempt >= max_tries:
            break
        wait = sleep_secs[min(attempt - 1, len(sleep_secs) - 1)]
        print(f"[WARN] waiting {wait:.0f}s before rebuild (attempt {attempt+1}/{max_tries})")
        time.sleep(wait)
        try:
            cur_env = env_builder()
        except BaseException as e2:
            print(f"[WARN] env_builder() failed: {e2}")
            last_err = Exception(str(e2))
    raise RuntimeError(f"robust_reset failed after {max_tries} attempts. Last: {last_err}")


@torch.no_grad()
def evaluate(
    env: CarlaReliableTransferEnv,
    env_builder: Callable[[], CarlaReliableTransferEnv],
    agent: SACAgent,
    episodes: int = 20,
    tag: str = "Town02_eval",
    cfg: Config = CFG,
) -> Tuple[Dict[str, object], CarlaReliableTransferEnv]:
    csv_path = os.path.join(cfg.result_dir, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "episode", "steps", "reward", "success", "goal", "collision", "off_road", "timeout", "reason",
                "progress_pct", "goal_dist", "goal_euclid", "mean_dL", "mean_heading", "min_ttc",
                "distance_km", "coll_per_km", "off_per_km", "to_per_km", "IS", "DS", "intervention_rate",
            ]
        )

    summary_path = os.path.join(cfg.result_dir, f"{tag}_summary.csv")
    totals = {"reward": [], "sr": 0, "dist_km": 0.0, "coll": 0, "off": 0, "to": 0, "ds": [], "is": [], "min_ttc": [], "intervention": []}

    cur_env = env
    for ep in range(1, episodes + 1):
        cur_env, obs, _ = robust_reset(
            cur_env,
            env_builder,
            npc_count=random.randint(cfg.npc_min, cfg.npc_max),
            goal_index=cfg.target_goal_index,
            max_tries=3,
        )

        done = False
        total = 0.0
        steps = 0
        dL_acc = 0.0
        head_acc = 0.0
        min_ttc = 999.0
        info: Dict[str, object] = {}

        while not done:
            action = agent.act(obs, deterministic=True)
            try:
                next_obs, _, terminated, truncated, info = cur_env.step(action)
            except BaseException as e:
                print(f"[WARN] env.step failed in episode {ep}: {e}")
                info = make_server_error_info(action, reason="server_error")
                next_obs = obs
                terminated, truncated = False, True
                done = True
                try:
                    cur_env.close()
                except BaseException:
                    pass
                try:
                    cur_env = env_builder()
                except BaseException as be:
                    print(f"[WARN] env rebuild after step failure: {be}")
            else:
                done = bool(terminated or truncated)

            applied_action = np.array(info.get("applied_action", action), dtype=np.float32)
            r, _ = recompute_agent_reward(agent, obs, applied_action, info, cfg=cfg)
            total += r
            steps += 1
            dL_acc += safe_float(info.get("dL", 0.0))
            head_acc += abs(safe_float(info.get("lane_head", 0.0)))
            min_ttc = min(min_ttc, safe_float(info.get("time_to_conflict", 999.0)))
            obs = next_obs

        mean_dL = dL_acc / max(1, steps)
        mean_head = head_acc / max(1, steps)
        dist_km = safe_float(info.get("distance_driven_m", 0.0)) / 1000.0
        coll = int(bool(info.get("collision", 0)))
        off = int(bool(info.get("off_road", 0)))
        timeout = int(bool(info.get("timeout", 0)))
        coll_per_km = coll / max(dist_km, 1e-6)
        off_per_km = off / max(dist_km, 1e-6)
        to_per_km = timeout / max(dist_km, 1e-6)
        iscore, _ = compute_infraction_scores(coll_per_km, off_per_km, to_per_km)
        ds = 100.0 * (safe_float(info.get("route_completion_pct", 0.0)) / 100.0) * iscore
        intervention_rate = safe_float(info.get("intervention_rate", 0.0))

        totals["reward"].append(total)
        totals["sr"] += int(bool(info.get("success", 0)))
        totals["dist_km"] += dist_km
        totals["coll"] += coll
        totals["off"] += off
        totals["to"] += timeout
        totals["ds"].append(ds)
        totals["is"].append(iscore)
        totals["min_ttc"].append(min_ttc)
        totals["intervention"].append(intervention_rate)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    ep, steps, f"{total:.2f}", int(bool(info.get("success", 0))), int(bool(info.get("goal_reached", 0))),
                    coll, off, timeout, str(info.get("term_reason", "")),
                    f"{safe_float(info.get('route_completion_pct', 0.0)):.3f}", f"{safe_float(info.get('goal_dist', 0.0)):.3f}",
                    f"{safe_float(info.get('goal_euclid', 0.0)):.3f}", f"{mean_dL:.3f}", f"{mean_head:.3f}", f"{min_ttc:.3f}",
                    f"{dist_km:.4f}", f"{coll_per_km:.4f}", f"{off_per_km:.4f}", f"{to_per_km:.4f}", f"{iscore:.4f}", f"{ds:.3f}", f"{intervention_rate:.4f}",
                ]
            )
        print(
            f"[{tag}] ep={ep:02d} R={total:.2f} steps={steps} success={int(bool(info.get('success', 0)))} "
            f"progress={safe_float(info.get('route_completion_pct', 0.0)):.1f}% DS={ds:.2f} intv={intervention_rate:.3f} reason={info.get('term_reason')}"
        )

    total_coll_per_km = totals["coll"] / max(totals["dist_km"], 1e-6)
    total_off_per_km = totals["off"] / max(totals["dist_km"], 1e-6)
    total_to_per_km = totals["to"] / max(totals["dist_km"], 1e-6)
    overall_sr = 100.0 * totals["sr"] / max(episodes, 1)

    summary = {
        "tag": tag,
        "episodes": int(episodes),
        "avg_reward": float(np.mean(totals["reward"])) if totals["reward"] else 0.0,
        "success_rate_pct": float(overall_sr),
        "avg_DS": float(np.mean(totals["ds"])) if totals["ds"] else 0.0,
        "avg_IS": float(np.mean(totals["is"])) if totals["is"] else 0.0,
        "avg_min_ttc": float(np.mean(totals["min_ttc"])) if totals["min_ttc"] else 999.0,
        "avg_intervention_rate": float(np.mean(totals["intervention"])) if totals["intervention"] else 0.0,
        "coll_per_km": float(total_coll_per_km),
        "off_per_km": float(total_off_per_km),
        "to_per_km": float(total_to_per_km),
        "total_dist_km": float(totals["dist_km"]),
        "episodes_csv": csv_path,
        "summary_csv": summary_path,
    }

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["avg_reward", "success_rate_pct", "avg_DS", "avg_IS", "avg_min_ttc", "avg_intervention_rate", "coll_per_km", "off_per_km", "to_per_km", "total_dist_km"])
        writer.writerow(
            [
                f"{summary['avg_reward']:.4f}",
                f"{summary['success_rate_pct']:.4f}",
                f"{summary['avg_DS']:.4f}",
                f"{summary['avg_IS']:.4f}",
                f"{summary['avg_min_ttc']:.4f}",
                f"{summary['avg_intervention_rate']:.4f}",
                f"{summary['coll_per_km']:.4f}",
                f"{summary['off_per_km']:.4f}",
                f"{summary['to_per_km']:.4f}",
                f"{summary['total_dist_km']:.4f}",
            ]
        )

    print(
        f"[{tag}] avgR={summary['avg_reward']:.2f} SR={summary['success_rate_pct']:.1f}% avgDS={summary['avg_DS']:.2f} "
        f"avgIS={summary['avg_IS']:.3f} intv={summary['avg_intervention_rate']:.3f} coll/km={summary['coll_per_km']:.4f} "
        f"off/km={summary['off_per_km']:.4f} to/km={summary['to_per_km']:.4f}"
    )
    return summary, cur_env


# ======================================================
# Training / Adaptation helpers
# ======================================================
def wait_for_carla_ready(host: str, port: int, timeout_s: float = 60.0, poll_interval: float = 2.0) -> None:
    """Poll until CARLA's TCP port accepts connections (server finished apply_settings)."""
    import socket
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError:
            pass
        time.sleep(poll_interval)


def make_target_cfg(base_cfg: Config) -> Config:
    """Relaxed routing config for target-map evaluation/adaptation."""
    import copy
    cfg = copy.copy(base_cfg)
    cfg.strict_goal_route = False
    cfg.allow_fallback_route = True
    cfg.enforce_route_length_for_fixed_goal = False
    # Only apply the 100 m floor if the user has NOT requested a custom length.
    # If --route-target-length was set (e.g. 500 m), preserve those values so
    # the longer route is maintained in evaluation.
    if base_cfg.route_target_length_m <= 210.0:
        cfg.min_route_length_m = 100.0
        cfg.candidate_goal_min_dist_m = 100.0
        cfg.route_soft_min_length_m = 100.0
        cfg.route_soft_max_length_m = 230.0
    else:
        # Custom length: keep base_cfg values (already scaled), just relax floors slightly
        cfg.min_route_length_m = base_cfg.min_route_length_m * 0.85
        cfg.candidate_goal_min_dist_m = base_cfg.candidate_goal_min_dist_m * 0.85
        cfg.route_soft_min_length_m = base_cfg.route_soft_min_length_m * 0.85
        cfg.route_soft_max_length_m = base_cfg.route_soft_max_length_m * 1.10
    cfg.target_goal_index = int(base_cfg.target_goal_index)
    return cfg


def make_env_builder(
    host: str,
    port: int,
    town_name: str,
    spawn_index: int,
    goal_index: int,
    weather_mode: str,
    cfg: Config,
) -> Callable[[], CarlaReliableTransferEnv]:
    return lambda: CarlaReliableTransferEnv(
        host=host,
        port=port,
        town_name=town_name,
        fixed_spawn_index=spawn_index,
        fixed_goal_index=goal_index,
        weather_mode=weather_mode,
        cfg=cfg,
    )


def sample_exploration_action(cfg: Config = CFG) -> np.ndarray:
    throttle = float(np.random.uniform(0.18, 0.62))
    brake = 0.0 if np.random.rand() < 0.90 else float(np.random.uniform(0.08, 0.32))
    if brake > 0.0:
        throttle = 0.0
    steer = float(np.random.uniform(-0.35, 0.35))
    return np.array([throttle, brake, steer], dtype=np.float32)


def collect_domain_batches(
    env_builder: Callable[[], CarlaReliableTransferEnv],
    agent: SACAgent,
    device: torch.device,
    batch_count: int,
    batch_size: int,
    goal_index: int,
    npc_min: int,
    npc_max: int,
    host: str = "localhost",
    port: int = 2200,
) -> List[Dict[str, torch.Tensor]]:
    if batch_count <= 0:
        return []
    env = env_builder()
    buf = ReplayBuffer(capacity=max(batch_count * batch_size * 2, batch_size + 1), cfg=agent.cfg)
    try:
        obs = None
        for _ in range(max(batch_count * batch_size * 3, batch_size)):
            if obs is None:
                env, obs, _ = robust_reset(env, env_builder,
                                           npc_count=random.randint(npc_min, npc_max),
                                           goal_index=goal_index, max_tries=5)
            action = sample_exploration_action(agent.cfg)
            try:
                next_obs, _, terminated, truncated, info = env.step(action)
            except BaseException as e:
                print(f"[WARN] collect_domain_batches step failed: {e}")
                obs = None
                continue
            reward, _ = recompute_agent_reward(agent, obs, action, info, cfg=agent.cfg)
            buf.add(obs, action, reward, next_obs, bool(terminated or truncated))
            obs = None if (terminated or truncated) else next_obs
            if len(buf) >= batch_count * batch_size:
                break
        batches: List[Dict[str, torch.Tensor]] = []
        if len(buf) >= batch_size:
            for _ in range(batch_count):
                batches.append(buf.sample(batch_size, device))
        return batches
    finally:
        try:
            env.close()
        except BaseException:
            pass
        print("[INFO] collect_domain_batches: waiting 25 s for CARLA to settle...")
        time.sleep(25.0)
        wait_for_carla_ready(host, port, timeout_s=60.0)
        print("[INFO] CARLA ready; proceeding to train_loop.")


def sample_holdout_batches_from_replay(
    replay: ReplayBuffer,
    device: torch.device,
    batch_count: int,
    batch_size: int,
) -> List[Dict[str, torch.Tensor]]:
    if len(replay) < batch_size or batch_count <= 0:
        return []
    return [replay.sample(batch_size, device) for _ in range(batch_count)]


def maybe_periodic_eval(
    agent: SACAgent,
    args: argparse.Namespace,
    cfg: Config,
    device: torch.device,
    step_idx: int,
    tag: str,
    train_town: str = "",
) -> None:
    eval_town = str(args.target_town).split("/")[-1].split(".")[0]
    if cfg.skip_cross_town_periodic_eval:
        norm = lambda t: t.lower().replace("_opt", "")
        if norm(train_town) != norm(eval_town):
            print(f"[EVAL step={step_idx}] skipping cross-town eval ({train_town}→{eval_town}).")
            return
    eval_cfg = make_target_cfg(cfg)
    eval_builder = make_env_builder(
        host=args.host, port=args.port, town_name=args.target_town,
        spawn_index=args.spawn_index, goal_index=eval_cfg.target_goal_index,
        weather_mode=eval_cfg.target_weather, cfg=eval_cfg,
    )
    env: Optional[CarlaReliableTransferEnv] = None
    try:
        env = eval_builder()
        summary, env = evaluate(env, eval_builder, agent,
                                episodes=int(args.eval_episodes),
                                tag=f"{tag}_step{step_idx}", cfg=eval_cfg)
        print(f"[EVAL step={step_idx}] SR={safe_float(summary['success_rate_pct']):.1f}% "
              f"DS={safe_float(summary['avg_DS']):.2f} IS={safe_float(summary['avg_IS']):.3f}")
    except BaseException as e:
        print(f"[WARN] periodic eval at step {step_idx} failed: {e}")
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        time.sleep(2.0)


def train_loop(
    agent: SACAgent,
    args: argparse.Namespace,
    cfg: Config,
    device: torch.device,
    env_builder: Callable[[], CarlaReliableTransferEnv],
    save_path: str,
    total_steps: int,
    goal_index: int,
    npc_min: int,
    npc_max: int,
    source_agent: Optional[SACAgent] = None,
    source_env_builder: Optional[Callable[[], CarlaReliableTransferEnv]] = None,
    source_goal_index: Optional[int] = None,
    source_npc_min: Optional[int] = None,
    source_npc_max: Optional[int] = None,
    max_episodes: Optional[int] = None,
) -> str:
    replay = ReplayBuffer(capacity=cfg.replay_size, cfg=cfg)
    source_replay: Optional[ReplayBuffer] = None
    if source_agent is not None and source_env_builder is not None:
        source_replay = ReplayBuffer(capacity=cfg.replay_size, cfg=cfg)

    _tl_host = getattr(args, "host", "localhost")
    _tl_port = getattr(args, "port", 2200)
    env: Optional[CarlaReliableTransferEnv] = None
    for _init_attempt in range(8):
        try:
            wait_for_carla_ready(_tl_host, _tl_port, timeout_s=30.0)
            env = env_builder()
            break
        except BaseException as e:
            wait = [10, 15, 20, 25, 30, 35, 40, 45][min(_init_attempt, 7)]
            print(f"[WARN] train_loop env init attempt {_init_attempt+1}/8 failed: {e}; retrying in {wait}s")
            time.sleep(wait)
    if env is None:
        raise RuntimeError("train_loop: could not create initial environment after 8 attempts")

    source_env: Optional[CarlaReliableTransferEnv] = None
    if source_env_builder is not None and source_agent is not None:
        for _sinit in range(4):
            try:
                source_env = source_env_builder()
                break
            except BaseException as e:
                wait = [5, 8, 12, 18][min(_sinit, 3)]
                print(f"[WARN] source_env init attempt {_sinit+1}/4 failed: {e}; retrying in {wait}s")
                time.sleep(wait)
    obs: Optional[Dict[str, np.ndarray]] = None
    source_obs: Optional[Dict[str, np.ndarray]] = None
    episode = 0
    episode_reward = 0.0
    env_steps = 0
    best_success_like = -1.0
    best_path = os.path.splitext(save_path)[0] + '_best.pt'
    last_calibration_step = -1
    try:
        while True:
            if obs is None and max_episodes is not None and episode >= max_episodes:
                break
            if env_steps >= total_steps:
                break

            if obs is None:
                env, obs, _ = robust_reset(
                    env,
                    env_builder,
                    npc_count=random.randint(npc_min, npc_max),
                    goal_index=goal_index,
                    max_tries=3,
                )
                episode += 1
                episode_reward = 0.0

            if env_steps < int(args.start_steps):
                action = sample_exploration_action(cfg)
            else:
                action = agent.act(obs, deterministic=False)

            next_obs, _, terminated, truncated, info = env.step(action)
            env_steps += 1
            reward, sigma_bar = recompute_agent_reward(agent, obs, action, info, cfg=cfg)
            info["sigma_bar"] = float(sigma_bar)
            done = bool(terminated or truncated)
            replay.add(obs, action, reward, next_obs, done)
            obs = None if done else next_obs
            episode_reward += reward

            if source_agent is not None and source_env is not None and source_replay is not None and source_env_builder is not None:
                if source_obs is None:
                    source_env, source_obs, _ = robust_reset(
                        source_env,
                        source_env_builder,
                        npc_count=random.randint(int(source_npc_min or npc_min), int(source_npc_max or npc_max)),
                        goal_index=int(source_goal_index if source_goal_index is not None else goal_index),
                        max_tries=3,
                    )
                source_action = source_agent.act(source_obs, deterministic=False)
                source_next_obs, _, source_terminated, source_truncated, source_info = source_env.step(source_action)
                source_reward, source_sigma_bar = recompute_agent_reward(source_agent, source_obs, source_action, source_info, cfg=cfg)
                source_info["sigma_bar"] = float(source_sigma_bar)
                source_done = bool(source_terminated or source_truncated)
                source_replay.add(source_obs, source_action, source_reward, source_next_obs, source_done)
                source_obs = None if source_done else source_next_obs

            if len(replay) >= cfg.batch_size and env_steps >= int(args.update_after):
                for _ in range(int(args.updates_per_step)):
                    batch = replay.sample(cfg.batch_size, device)
                    source_batch = None
                    if source_agent is not None and source_replay is not None and len(source_replay) >= cfg.batch_size:
                        source_batch = source_replay.sample(cfg.batch_size, device)
                    stats = agent.update(batch, source_agent=source_agent if source_batch is not None else None, source_batch=source_batch)
                if cfg.debug_mode and agent.training_steps % max(1, cfg.debug_step_freq * 5) == 0:
                    print(
                        f"[TRAIN] step={agent.training_steps} critic={safe_float(stats.get('critic_loss')):.4f} "
                        f"actor={safe_float(stats.get('actor_loss')):.4f} alpha={safe_float(stats.get('alpha')):.4f} "
                        f"sigma={safe_float(stats.get('sigma_bar')):.4f} beta0={cfg.beta0:.2f}"
                    )

            if (
                len(replay) >= cfg.batch_size
                and env_steps > 0
                and env_steps % max(1, int(cfg.calibrate_every_steps)) == 0
                and env_steps != last_calibration_step
            ):
                holdout_batches = sample_holdout_batches_from_replay(replay, device, batch_count=int(cfg.calib_num_batches), batch_size=cfg.batch_size)
                if holdout_batches:
                    cal_stats = agent.fit_uncertainty_calibrator(holdout_batches)
                    beta_stats: Dict[str, float] = {}
                    if bool(cfg.auto_beta0_selection):
                        beta_stats = agent.select_beta0_from_holdout(holdout_batches)
                    last_calibration_step = env_steps
                    if cfg.debug_mode:
                        print(
                            f"[CAL] step={env_steps} fitted={int(cal_stats.get('calib_fitted', 0.0))} "
                            f"center={safe_float(cal_stats.get('calib_center')):.4f} scale={safe_float(cal_stats.get('calib_scale')):.4f} "
                            f"beta0={safe_float(beta_stats.get('beta0_selected', cfg.beta0)):.2f}"
                        )

            if done:
                success_like = 1.0 if bool(info.get('success', False)) else safe_float(info.get('route_completion_pct', 0.0)) / 100.0
                if success_like > best_success_like:
                    best_success_like = success_like
                    agent.save(best_path)
                if cfg.debug_mode:
                    print(
                        f"[EP] ep={episode} steps={safe_float(info.get('steps', 0)):.0f} reward={episode_reward:.2f} "
                        f"success={int(bool(info.get('success', False)))} progress={safe_float(info.get('route_completion_pct', 0.0)):.1f}% "
                        f"reason={info.get('term_reason', '')}"
                    )

            if env_steps > 0 and env_steps % int(args.save_every_steps) == 0:
                agent.save(save_path)
                print(f"[OK] Saved checkpoint: {save_path}")

            if env_steps > 0 and env_steps % int(args.eval_every_steps) == 0:
                train_town = getattr(env, "town_name", "")
                agent.eval()
                try:
                    maybe_periodic_eval(agent, args, cfg, device, env_steps,
                                        tag=os.path.splitext(os.path.basename(save_path))[0],
                                        train_town=train_town)
                except BaseException as e:
                    print(f"[WARN] maybe_periodic_eval raised: {e}")
                agent.train()

        holdout_batches = sample_holdout_batches_from_replay(replay, device, batch_count=int(cfg.calib_num_batches), batch_size=cfg.batch_size)
        if holdout_batches:
            agent.fit_uncertainty_calibrator(holdout_batches)
            if bool(cfg.auto_beta0_selection):
                agent.select_beta0_from_holdout(holdout_batches)
        agent.save(save_path)
        print(f"[OK] Final checkpoint saved: {save_path}")
        if os.path.exists(best_path):
            print(f"[OK] Best checkpoint saved : {best_path}")
        return save_path
    finally:
        try:
            env.close()
        except BaseException:
            pass
        if source_env is not None:
            try:
                source_env.close()
            except BaseException:
                pass


def run_source_training(args: argparse.Namespace, cfg: Config, device: torch.device) -> None:
    # Use random spawn (-1) for source training unless the user explicitly fixed one.
    # Spawn diversity is critical: training log shows 97.3% collision rate when
    # spawn is fixed to index 0 — the policy overfits to one road segment and
    # hits the same NPC deterministically after alpha collapses at ~18k steps.
    train_spawn_index = args.spawn_index if args.spawn_index >= 0 else -1
    train_builder = make_env_builder(
        host=args.host,
        port=args.port,
        town_name=args.train_town,
        spawn_index=train_spawn_index,
        goal_index=cfg.fixed_goal_index,
        weather_mode=cfg.fixed_weather,
        cfg=cfg,
    )
    agent = SACAgent(device=device, cfg=cfg)
    if int(args.maml_warmup_batches) > 0:
        warm_batches = collect_domain_batches(
            train_builder, agent, device,
            batch_count=int(args.maml_warmup_batches),
            batch_size=cfg.batch_size,
            goal_index=cfg.fixed_goal_index,
            npc_min=cfg.train_npc_min,
            npc_max=cfg.train_npc_max,
            host=args.host,
            port=args.port,
        )
        if warm_batches:
            meta_stats = agent.maml_style_initialize(warm_batches)
            print(f"[OK] Applied MAML-style initialization: {meta_stats}")
    save_path = resolve_existing_path(os.path.join(cfg.model_dir, 'source_agent.pt'))
    train_loop(
        agent=agent,
        args=args,
        cfg=cfg,
        device=device,
        env_builder=train_builder,
        save_path=save_path,
        total_steps=int(args.train_steps),
        goal_index=cfg.fixed_goal_index,
        npc_min=cfg.train_npc_min,
        npc_max=cfg.train_npc_max,
        source_agent=None,
    )


def run_target_adaptation(args: argparse.Namespace, cfg: Config, device: torch.device) -> None:
    source_ckpt = resolve_existing_path(args.source_checkpoint or args.checkpoint)
    if not os.path.exists(source_ckpt):
        raise FileNotFoundError(f'Source checkpoint not found: {source_ckpt}')
    source_agent = SACAgent(device=device, cfg=cfg)
    source_agent.load(source_ckpt)
    source_agent.eval()
    target_cfg = make_target_cfg(cfg)
    target_agent = SACAgent(device=device, cfg=target_cfg)
    target_agent.load(source_ckpt)
    target_agent.train()
    adapt_builder = make_env_builder(
        host=args.host, port=args.port, town_name=args.target_town,
        spawn_index=args.spawn_index, goal_index=target_cfg.target_goal_index,
        weather_mode=target_cfg.target_weather, cfg=target_cfg,
    )
    source_builder = make_env_builder(
        host=args.host, port=args.port, town_name=args.train_town,
        spawn_index=args.spawn_index, goal_index=cfg.fixed_goal_index,
        weather_mode=cfg.fixed_weather, cfg=cfg,
    )
    if int(args.maml_warmup_batches) > 0:
        warm_batches = collect_domain_batches(
            adapt_builder, target_agent, device,
            batch_count=int(args.maml_warmup_batches), batch_size=target_cfg.batch_size,
            goal_index=target_cfg.target_goal_index,
            npc_min=target_cfg.npc_min, npc_max=target_cfg.npc_max,
            host=args.host, port=args.port,
        )
        if warm_batches:
            meta_stats = target_agent.maml_style_initialize(warm_batches)
            print(f"[OK] Applied target-domain MAML-style initialization: {meta_stats}")
    save_path = resolve_existing_path(os.path.join(cfg.model_dir, 'target_agent.pt'))
    train_loop(
        agent=target_agent, args=args, cfg=target_cfg, device=device,
        env_builder=adapt_builder, save_path=save_path,
        total_steps=int(args.adapt_steps), goal_index=target_cfg.target_goal_index,
        npc_min=target_cfg.npc_min, npc_max=target_cfg.npc_max,
        source_agent=source_agent, source_env_builder=source_builder,
        source_goal_index=cfg.fixed_goal_index,
        source_npc_min=cfg.train_npc_min, source_npc_max=cfg.train_npc_max,
        max_episodes=int(args.adapt_episodes),
    )


def run_target_policy_learning(args: argparse.Namespace, cfg: Config, device: torch.device) -> None:
    target_cfg = make_target_cfg(cfg)
    policy_builder = make_env_builder(
        host=args.host, port=args.port, town_name=args.target_town,
        spawn_index=args.spawn_index, goal_index=target_cfg.target_goal_index,
        weather_mode=target_cfg.target_weather, cfg=target_cfg,
    )
    agent = SACAgent(device=device, cfg=target_cfg)
    save_path = resolve_existing_path(os.path.join(cfg.model_dir, 'target_policy_agent.pt'))
    train_loop(
        agent=agent, args=args, cfg=target_cfg, device=device,
        env_builder=policy_builder, save_path=save_path,
        total_steps=int(args.adapt_steps), goal_index=target_cfg.target_goal_index,
        npc_min=target_cfg.npc_min, npc_max=target_cfg.npc_max,
        source_agent=None, max_episodes=int(args.adapt_episodes),
    )


# ======================================================
# CLI / main
# ======================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA 0.9.15 – Reliable E2E Policy Transfer with DRL (paper-aligned, 200 m route)")
    p.add_argument("--mode", type=str, default="eval", choices=["eval", "train", "adapt", "policy"],
                   help="eval=checkpoint evaluation, train=source training (Town10HD→200m route), "
                        "adapt=target-domain few-shot adaptation (Eq.16-17), policy=target-only SAC without transfer")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2200)
    p.add_argument("--tm-port", type=int, default=CFG.tm_port)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--fps", type=int, default=CFG.fps)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--spawn-index", type=int, default=0)
    p.add_argument("--train-town", type=str, default="Town10HD_Opt")
    p.add_argument("--target-town", type=str, default="Town02")
    p.add_argument("--source-weather", type=str, default="night_rain_fog", choices=["night_rain_fog", "mixed", "default"])
    p.add_argument("--target-weather", type=str, default="mixed", choices=["night_rain_fog", "mixed", "default"])
    p.add_argument("--train-goal-index", type=int, default=40)
    p.add_argument("--target-goal-index", type=int, default=-1,
                   help="-1 = auto-route (safe for Town02)")
    p.add_argument("--npc-min", type=int, default=CFG.npc_min)
    p.add_argument("--npc-max", type=int, default=CFG.npc_max)
    p.add_argument("--train-npc-min", type=int, default=CFG.train_npc_min)
    p.add_argument("--train-npc-max", type=int, default=CFG.train_npc_max)
    p.add_argument("--checkpoint", type=str, default=os.path.join(CFG.model_dir, "source_agent.pt"))
    p.add_argument("--source-checkpoint", type=str, default="", help="source checkpoint for adaptation; defaults to --checkpoint")
    p.add_argument("--out-dir", type=str, default=CFG.out_dir)
    p.add_argument("--train-steps", type=int, default=500000,   # paper: 5×10^5 steps
                   help="total environment steps for source training")
    p.add_argument("--adapt-steps", type=int, default=50000)
    p.add_argument("--adapt-episodes", type=int, default=CFG.adapt_episodes)
    p.add_argument("--start-steps", type=int, default=5000)
    p.add_argument("--update-after", type=int, default=2048)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--save-every-steps", type=int, default=10000)
    p.add_argument("--eval-every-steps", type=int, default=10000)
    p.add_argument("--maml-warmup-batches", type=int, default=2)
    p.add_argument("--maml-inner-steps", type=int, default=CFG.maml_inner_steps)
    p.add_argument("--calibrate-every-steps", type=int, default=CFG.calibrate_every_steps)
    p.add_argument("--calib-num-batches", type=int, default=CFG.calib_num_batches)
    p.add_argument("--disable-auto-beta0", action="store_true")
    p.add_argument("--beta0-candidates", type=str, default=",".join(str(x) for x in CFG.beta0_candidates))
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-step-freq", type=int, default=CFG.debug_step_freq)
    p.add_argument("--no-follow-ego-view", action="store_true")
    p.add_argument("--no-rendering", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--use-safety-shield", action="store_true", help="enable extra rule-based route/safety intervention; disabled by default for paper-faithful learning/evaluation")
    p.add_argument("--route-target-length", type=float, default=0.0,
                   help="Override route target length in metres (0 = use default 200m). "
                        "Set to 500 for longer trajectory plots. Scales all related route "
                        "length thresholds proportionally.")
    return p.parse_args()


def make_runtime_cfg(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.tm_port = int(args.tm_port)
    cfg.seed = int(args.seed)
    cfg.fps = int(args.fps)
    cfg.debug_mode = bool(args.debug)
    cfg.debug_step_freq = int(args.debug_step_freq)
    cfg.mode_name = str(args.mode)
    cfg.follow_ego_view = not bool(args.no_follow_ego_view)
    cfg.no_rendering_mode = bool(args.no_rendering)
    cfg.fixed_goal_index = int(args.train_goal_index)
    cfg.target_goal_index = int(args.target_goal_index)
    cfg.npc_min = int(args.npc_min)
    cfg.npc_max = int(args.npc_max)
    cfg.train_npc_min = int(args.train_npc_min)
    cfg.train_npc_max = int(args.train_npc_max)
    cfg.fixed_weather = str(args.source_weather)
    cfg.target_weather = str(args.target_weather)
    cfg.use_safety_shield = bool(args.use_safety_shield)
    cfg.maml_inner_steps = int(args.maml_inner_steps)
    cfg.calibrate_every_steps = int(args.calibrate_every_steps)
    cfg.calib_num_batches = int(args.calib_num_batches)
    cfg.auto_beta0_selection = not bool(args.disable_auto_beta0)
    beta0_candidates = []
    for part in str(args.beta0_candidates).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            beta0_candidates.append(float(part))
        except Exception:
            continue
    if beta0_candidates:
        cfg.beta0_candidates = tuple(beta0_candidates)
        cfg.beta0 = float(beta0_candidates[-1])
    cfg.adapt_episodes = int(args.adapt_episodes)
    cfg.out_dir = resolve_output_dir(str(args.out_dir))
    ensure_dirs(cfg)

    # Route length override — scales all related thresholds proportionally.
    # Example: --route-target-length 500 sets a ~500 m route for evaluation
    # which produces longer, more visually informative trajectory figures.
    _rtl = float(getattr(args, "route_target_length", 0.0))
    if _rtl > 50.0:
        _scale = _rtl / 200.0          # ratio vs default 200 m
        cfg.route_target_length_m          = _rtl
        cfg.route_target_tolerance_m       = max(15.0, 12.0 * _scale)
        cfg.route_soft_min_length_m        = _rtl * 0.94
        cfg.route_soft_max_length_m        = _rtl * 1.08
        cfg.min_route_length_m             = _rtl * 0.90
        cfg.candidate_goal_min_dist_m      = _rtl * 0.90
        cfg.candidate_goal_max_tries       = max(80, int(80 * _scale))
        cfg.max_reset_start_progress_pct   = min(10.0, 20.0 / _scale)
        print(f"[CFG] route_target_length overridden to {_rtl:.0f} m "
              f"(soft window {cfg.route_soft_min_length_m:.0f}–{cfg.route_soft_max_length_m:.0f} m)")

    return cfg


def main() -> None:
    args = parse_args()
    cfg = make_runtime_cfg(args)
    set_global_seed(cfg.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"GlobalRoutePlanner available: {GlobalRoutePlanner is not None}")

    print(f"Python: {sys.version.split()[0]}")
    src_goal_txt = "auto" if int(cfg.fixed_goal_index) < 0 else str(cfg.fixed_goal_index)
    tgt_goal_txt = "auto" if int(cfg.target_goal_index) < 0 else str(cfg.target_goal_index)
    print(f"Source town: {args.train_town} | goal={src_goal_txt} | NPC=[{cfg.train_npc_min}, {cfg.train_npc_max}]")
    print(f"Target town: {args.target_town} | goal={tgt_goal_txt} | NPC=[{cfg.npc_min}, {cfg.npc_max}]")
    print(f"Weather: source={cfg.fixed_weather} target={cfg.target_weather}")
    print(f"Safety shield: {cfg.use_safety_shield}")
    print(f"beta0 candidates: {cfg.beta0_candidates} | auto-select={cfg.auto_beta0_selection}")
    print(f"Follow ego view: {cfg.follow_ego_view}")
    print(f"No rendering: {cfg.no_rendering_mode}")
    print(f"Output dir: {cfg.out_dir}")
    print(f"Auto-selected route target length: {cfg.route_target_length_m:.0f}m ± {cfg.route_target_tolerance_m:.0f}m")
    print(f"Train steps: {args.train_steps} | Adapt steps cap: {args.adapt_steps} | Adapt episodes: {args.adapt_episodes}")

    if args.mode == "train":
        run_source_training(args, cfg, device)
        return

    if args.mode == "adapt":
        run_target_adaptation(args, cfg, device)
        return

    if args.mode == "policy":
        run_target_policy_learning(args, cfg, device)
        return

    checkpoint = resolve_existing_path(args.checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Checkpoint: {checkpoint}")
    print(f"Episodes: {args.eval_episodes}")

    agent = SACAgent(device=device, cfg=cfg)
    agent.load(checkpoint)
    print("[OK] Loaded checkpoint.")

    eval_cfg = make_target_cfg(cfg)
    env: Optional[CarlaReliableTransferEnv] = None
    try:
        tag = f"{args.target_town}_zeroshot_{os.path.splitext(os.path.basename(checkpoint))[0]}"
        print("=" * 90)
        eval_goal_txt = "auto" if int(eval_cfg.target_goal_index) < 0 else str(eval_cfg.target_goal_index)
        print(f"Evaluating on {args.target_town} | goal={eval_goal_txt} | weather={eval_cfg.target_weather}")
        env_builder = make_env_builder(
            host=args.host, port=args.port, town_name=args.target_town,
            spawn_index=args.spawn_index, goal_index=eval_cfg.target_goal_index,
            weather_mode=eval_cfg.target_weather, cfg=eval_cfg,
        )
        env = env_builder()
        print(f"Resolved target env town: {env.town_name}")
        summary, env = evaluate(env, env_builder, agent, episodes=args.eval_episodes, tag=tag, cfg=eval_cfg)

        print("=" * 90)
        print(
            f"Final summary for {env.town_name}: SR={safe_float(summary['success_rate_pct']):.1f}% "
            f"DS={safe_float(summary['avg_DS']):.2f} IS={safe_float(summary['avg_IS']):.3f} "
            f"coll/km={safe_float(summary['coll_per_km']):.4f} off/km={safe_float(summary['off_per_km']):.4f} "
            f"to/km={safe_float(summary['to_per_km']):.4f}"
        )
        print(f"Episodes CSV: {summary['episodes_csv']}")
        print(f"Summary CSV : {summary['summary_csv']}")
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass


if __name__ == "__main__":
    main()
