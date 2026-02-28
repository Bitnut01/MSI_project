import math
import numpy as np
import random
from typing import Dict, Tuple, List, Any
import heapq

from .strategy import StrategyType
from .observer import BattlefieldObserver
from .state import *


from pydantic import BaseModel

ROT_ANG = 30.0
BARREL_SPIN_RATE = 15.0

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

class Commander():
    
    def __init__(self):
        self.on_bad_terrain: bool = False



# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================        
    

def get_best_ammo(my_tank: Dict[str, Any]) -> str:

    ammo_data = my_tank.get("ammo", {})
    if not ammo_data:
        return "DEFAULT"
    return max(ammo_data, key=lambda k: ammo_data[k].get("count", 0))

def get_heading_to_pos(my_tank: Dict[str, Any], target_pos: Dict[str, float]) -> float:

    my_angle = my_tank.get("heading", 0.0) 
    dx = target_pos["x"] - my_tank["position"]["x"]
    dy = target_pos["y"] - my_tank["position"]["y"]
    target_angle = math.degrees(math.atan2(dy, dx))
    
    diff = target_angle - my_angle
    return (diff + 180) % 360 - 180


def _map_cfg_from_observer(observer: BattlefieldObserver) -> MapConfig:
    w = 200.0
    h = 200.0

    return MapConfig(width=w, height=h, margin=5.0)


def _rec_cfg_default() -> RecoveryConfig:
    return RecoveryConfig(
        backup_ticks=3,
        rotate_ticks=3,
        backup_speed=-5.0,
        forward_speed=5.0,
        stuck_eps=0.01,
        stuck_patience_ticks=50,
        border_patience_ticks=10,
    )


def _get_escape_target(
    summary: Dict[str, Any],
    observer: BattlefieldObserver,
    *,
    retreat_dist: float,
) -> Tuple[float, float, bool]:

    nearest = summary["radar"]["nearest_enemy"]
    my_pos = summary["self"]["pos"]
    mx = float(my_pos["x"])
    my = float(my_pos["y"])

    map_cfg = _map_cfg_from_observer(observer)

    if not nearest:
        tx = _clamp(mx, map_cfg.margin, map_cfg.width - map_cfg.margin)
        ty = _clamp(my, map_cfg.margin, map_cfg.height - map_cfg.margin)
        return tx, ty, False

    enemy_pos = nearest["tank_data"]["position"]
    ex = float(enemy_pos["x"])
    ey = float(enemy_pos["y"])

    vx = mx - ex
    vy = my - ey
    n = math.hypot(vx, vy)

    if n < 1e-6:
        ang = math.radians(random.uniform(0.0, 360.0))
        vx, vy = math.cos(ang), math.sin(ang)
        n = 1.0

    vx /= n
    vy /= n

    tx = mx + vx * float(retreat_dist)
    ty = my + vy * float(retreat_dist)

    tx = _clamp(tx, map_cfg.margin, map_cfg.width - map_cfg.margin)
    ty = _clamp(ty, map_cfg.margin, map_cfg.height - map_cfg.margin)

    return tx, ty, True

def _wrap_angle_deg(a: float) -> float:

    return (a + 180.0) % 360.0 - 180.0


def _clamp_rotation_delta(delta_deg: float, max_step: float) -> float:
    return max(-max_step, min(delta_deg, max_step))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _get_top_speed(my_tank: Any, default: float = 4.0) -> float:

    if isinstance(my_tank, dict):
        return float(my_tank.get("_top_speed", default))
    return float(getattr(my_tank, "_top_speed", default))


# ============================================================================
# TAKTYKI DLA POSZCZEGÓLNYCH STRATEGII
# ============================================================================

def tactic_attack(
    summary: Dict[str, Any],
    observer: BattlefieldObserver,
    state: IterState,
) -> ActionCommand:


    def _norm_angle(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0

    def _angle_diff(target: float, current: float) -> float:
        return _norm_angle(target - current)

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _clamp_rotation_delta(delta: float, max_abs: float) -> float:
        return _clamp(delta, -abs(max_abs), abs(max_abs))

    def _as_enum_name(v: Any) -> Optional[str]:
        if v is None:
            return None
        name = getattr(v, "name", None) 
        if isinstance(name, str):
            return name
        if isinstance(v, str):

            return v.split(".")[-1]
        return str(v)

    def _ammo_cmd_value(v: Any) -> Optional[str]:
        name = _as_enum_name(v)
        if name is None:
            return None

        return f"AmmoType.{name}"

    nearest = summary.get("radar", {}).get("nearest_enemy")
    ammo_to_load = get_best_ammo(observer.my_tank)

    rot_ang = float(globals().get("ROT_ANG", 30.0))
    barrel_spin_fallback = float(globals().get("BARREL_SPIN_RATE", 90.0))


    if nearest is None:
        state.set_forward(speed=5.0)
        speed, heading_rot = state.motion_step(
            summary,
            rot_ang=rot_ang,
            map_cfg=_map_cfg_from_observer(observer),
            rec_cfg=_rec_cfg_default(),
            enable_border=True,
            enable_obstacle=True,
            enable_stuck=True,
            enable_terrein=False,
        )
        return ActionCommand(
            barrel_rotation_angle=0.0,
            heading_rotation_angle=heading_rot,
            move_speed=speed,
            ammo_to_load=_ammo_cmd_value(ammo_to_load),
            should_fire=False,
        )

    my_pos = summary["self"]["pos"]
    my_x = float(my_pos["x"])
    my_y = float(my_pos["y"])
    my_heading = float(summary["self"]["heading"])
    my_barrel = float(summary["self"]["barrel_angle"])

    enemy_pos = nearest["tank_data"]["position"]
    ex = float(enemy_pos["x"])
    ey = float(enemy_pos["y"])

    dx = ex - my_x
    dy = ey - my_y
    dist_to_enemy = float(math.hypot(dx, dy))


    target_heading_deg = float(math.degrees(math.atan2(dy, dx)))
    heading_diff = _angle_diff(target_heading_deg, my_heading)


    err_abs = _angle_diff(target_heading_deg, my_barrel)


    desired_rel = _angle_diff(target_heading_deg, my_heading)
    err_rel = _angle_diff(desired_rel, my_barrel)

    barrel_err = err_rel if abs(err_rel) < abs(err_abs) else err_abs

    barrel_spin = float(
        observer.my_tank.get("_barrel_spin_rate", barrel_spin_fallback)
    )

    if abs(barrel_err) < 1.5:
        barrel_rot = 0.0
    else:
        barrel_rot = _clamp_rotation_delta(barrel_err * 0.6, barrel_spin)

    weapon_range = float(observer.ballistics.get_range(observer.my_tank))
    
    if weapon_range <= 0.0:
        weapon_range = 8.0  # fallback

    optimal_dist = 0.9 * weapon_range
    top_speed = float(_get_top_speed(observer.my_tank, 5.0))

    if dist_to_enemy > optimal_dist:
        desired_speed = top_speed
    elif dist_to_enemy < (optimal_dist * 0.4):
        desired_speed = -top_speed * 0.8
    else:
        err_dist = dist_to_enemy - optimal_dist
        desired_speed = _clamp(err_dist * 0.5, -top_speed, top_speed)


    if abs(heading_diff) > 45.0:
        desired_speed *= 0.1

    if dist_to_enemy > 1e-6:
        ux = (my_x - ex) / dist_to_enemy
        uy = (my_y - ey) / dist_to_enemy
        gx = ex + ux * optimal_dist
        gy = ey + uy * optimal_dist
    else:
        gx, gy = my_x, my_y

    state.set_goto(x=gx, y=gy, speed=desired_speed, stop_radius=3.0)

    speed, heading_rot = state.motion_step(
        summary,
        rot_ang=rot_ang,
        map_cfg=_map_cfg_from_observer(observer),
        rec_cfg=_rec_cfg_default(),
        enable_border=True,
        enable_obstacle=True,
        enable_stuck=True,
        enable_terrein=False,
    )

    heading_spin = float(observer.my_tank.get("_heading_spin_rate", rot_ang))
    face_enemy_rot = _clamp_rotation_delta(heading_diff * 0.7, heading_spin)

    close_combat = dist_to_enemy <= (weapon_range * 1.2)
    obstacle_ahead = bool(summary["self"].get("obstacle_ahead", False))
    if close_combat and not obstacle_ahead:
        heading_rot = (0.3 * float(heading_rot)) + (0.7 * float(face_enemy_rot))


    ammo_loaded_name = _as_enum_name(observer.my_tank.get("ammo_loaded"))
    want_ammo_name = _as_enum_name(ammo_to_load)

    clear_line = observer.ballistics.is_line_of_fire_clear(
        summary["self"]["pos"],
        nearest["tank_data"]["position"],
        observer.radar.allies,
    )

    ready = bool(summary["self"].get("is_ready", False))
    aim_ok = abs(barrel_err) <= 5

    correct_ammo = (
        want_ammo_name is None
        or ammo_loaded_name is None
        or ammo_loaded_name == want_ammo_name
    )

    should_fire = bool(ready and aim_ok and correct_ammo)
    
    return ActionCommand(
        barrel_rotation_angle=float(barrel_rot),
        heading_rotation_angle=float(heading_rot),
        move_speed=float(speed),
        ammo_to_load=_ammo_cmd_value(ammo_to_load),
        should_fire=should_fire,
    )

def tactic_flee(summary: Dict[str, Any], observer: BattlefieldObserver, state: IterState) -> ActionCommand:
    ammo_to_load = get_best_ammo(observer.my_tank)
    map_cfg = _map_cfg_from_observer(observer)
    rec_cfg = _rec_cfg_default()

    nearest = summary["radar"]["nearest_enemy"]

    def _apply_desired_mode_now() -> None:
        if state.motion_mode not in (MotionMode.BACKUP, MotionMode.ROTATE):
            state.motion_mode = state.desired_mode

    if nearest is None:
        state.goto_x = None
        state.goto_y = None
        state.set_forward(speed=0.0)
        _apply_desired_mode_now()
        return ActionCommand(
            barrel_rotation_angle=ROT_ANG,
            heading_rotation_angle=0.0,
            move_speed=0.0,
            ammo_to_load=ammo_to_load,
            should_fire=False,
        )

    my_pos = summary["self"]["pos"]
    mx = float(my_pos["x"])
    my = float(my_pos["y"])
    heading_deg = float(summary["self"]["heading"])

    enemy_pos = nearest["tank_data"]["position"]
    ex = float(enemy_pos["x"])
    ey = float(enemy_pos["y"])

    vx = mx - ex
    vy = my - ey
    n = math.hypot(vx, vy)
    if n < 1e-6:
        ang = math.radians(random.uniform(0.0, 360.0))
        vx, vy = math.cos(ang), math.sin(ang)
        n = 1.0
    vx /= n
    vy /= n

    dist_enemy = float(math.hypot(ex - mx, ey - my))
    retreat_dist = float(_clamp(35.0 + (25.0 - dist_enemy) * 1.5, 35.0, 90.0))

    base_ang = math.atan2(vy, vx)
    offsets = [
        0.0,
        math.radians(30.0),
        -math.radians(30.0),
        math.radians(60.0),
        -math.radians(60.0),
    ]

    def border_clearance(x: float, y: float) -> float:
        return min(
            x - map_cfg.margin,
            (map_cfg.width - map_cfg.margin) - x,
            y - map_cfg.margin,
            (map_cfg.height - map_cfg.margin) - y,
        )

    best_tx, best_ty = mx + vx * retreat_dist, my + vy * retreat_dist
    best_score = -1e18
    for off in offsets:
        ax = math.cos(base_ang + off)
        ay = math.sin(base_ang + off)
        tx = mx + ax * retreat_dist
        ty = my + ay * retreat_dist
        tx = _clamp(tx, map_cfg.margin, map_cfg.width - map_cfg.margin)
        ty = _clamp(ty, map_cfg.margin, map_cfg.height - map_cfg.margin)

        score = math.hypot(tx - ex, ty - ey) + 0.25 * border_clearance(tx, ty)
        if score > best_score:
            best_score = score
            best_tx, best_ty = tx, ty

    to_enemy_x = ex - mx
    to_enemy_y = ey - my
    h = math.radians(heading_deg)
    forward_x = math.cos(h)
    forward_y = math.sin(h)
    forward_dot_to_enemy = forward_x * to_enemy_x + forward_y * to_enemy_y

    ret_ang_deg = math.degrees(math.atan2(best_ty - my, best_tx - mx))
    err_ret = _wrap_angle_deg(ret_ang_deg - heading_deg)

    top_speed = 5.0
    if dist_enemy < 25.0 and (forward_dot_to_enemy > 0.0 or abs(err_ret) > 80.0):
        cmd_speed = 0.0
    else:
        cmd_speed = top_speed

    state.set_goto(x=best_tx, y=best_ty, speed=cmd_speed, stop_radius=2.0)
    _apply_desired_mode_now()

    speed, heading_rot = state.motion_step(
        summary,
        rot_ang=ROT_ANG,
        map_cfg=map_cfg,
        rec_cfg=rec_cfg,
        enable_border=True,
        enable_obstacle=True,
        enable_stuck=True,
        enable_terrein=True,
    )

    barrel_rot = _clamp_rotation_delta(
        float(summary["tactical"]["rotation_to_target"]),
        BARREL_SPIN_RATE,
    )

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=float(heading_rot),
        move_speed=float(speed),
        ammo_to_load=ammo_to_load,
        should_fire=bool(summary["tactical"]["can_fire"]),)

def tactic_save(summary: Dict[str, Any], observer: BattlefieldObserver, state: IterState) -> ActionCommand:
    
    ammo_to_load = get_best_ammo(observer.my_tank)
    map_cfg = _map_cfg_from_observer(observer)
    rec_cfg = _rec_cfg_default()

    def _apply_desired_mode_now() -> None:

        if state.motion_mode not in (MotionMode.BACKUP, MotionMode.ROTATE):
            state.motion_mode = state.desired_mode

    terrain_damage = int(summary["self"].get("terrain_damage", 0))


    if terrain_damage <= 0:
        state.goto_x = None
        state.goto_y = None

        # Wymuś stop
        state.set_forward(speed=0.0)
        _apply_desired_mode_now()

        state.motion_mode = MotionMode.FORWARD
        state.motion_ticks_left = 0

        return ActionCommand(
            barrel_rotation_angle=ROT_ANG,  # skan
            heading_rotation_angle=0.0,
            move_speed=0.0,
            ammo_to_load=ammo_to_load,
            should_fire=False,
        )

    my_pos = summary["self"]["pos"]
    mx = float(my_pos["x"])
    my = float(my_pos["y"])

    need_new_target = state.goto_x is None or state.goto_y is None
    if not need_new_target:
        dist_to_target = float(
            math.hypot(float(state.goto_x) - mx, float(state.goto_y) - my)
        )

        if dist_to_target <= max(2.0, state.goto_stop_radius):
            need_new_target = True

    if need_new_target:

        ang = math.radians(random.uniform(0.0, 360.0))
        step = random.uniform(20.0, 35.0)
        tx = mx + math.cos(ang) * step
        ty = my + math.sin(ang) * step

        tx = _clamp(tx, map_cfg.margin, map_cfg.width - map_cfg.margin)
        ty = _clamp(ty, map_cfg.margin, map_cfg.height - map_cfg.margin)

        state.set_goto(x=tx, y=ty, speed=4.0, stop_radius=2.0)

    _apply_desired_mode_now()


    speed, heading_rot = state.motion_step(
        summary,
        rot_ang=ROT_ANG,
        map_cfg=map_cfg,
        rec_cfg=rec_cfg,
        enable_border=True,
        enable_obstacle=True,
        enable_stuck=True,
        enable_terrein=False,
    )

    return ActionCommand(
        barrel_rotation_angle=ROT_ANG,  # skan w trakcie schodzenia
        heading_rotation_angle=float(heading_rot),
        move_speed=float(speed),
        ammo_to_load=ammo_to_load,
        should_fire=False,
    )

def tactic_search(summary, observer, state: IterState) -> ActionCommand:
    
    nearest = summary["radar"]["nearest_enemy"]

    if nearest:
        barrel_rot = _clamp_rotation_delta(
            float(summary["tactical"]["rotation_to_target"]),
            BARREL_SPIN_RATE,
        )
    else:
        barrel_rot = 30.0

    hp = float(summary["self"]["hp_pct"])

    state.set_forward(speed=5.0)

    speed, heading_rot = state.motion_step(
        summary,
        rot_ang=ROT_ANG,
        map_cfg=_map_cfg_from_observer(observer),
        rec_cfg=_rec_cfg_default(),
        enable_border=True,
        enable_obstacle=True,
        enable_stuck=True,
        enable_terrein=True if hp < 15.0 else False,
    )

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=heading_rot,
        move_speed=speed,
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=False,
    )

def tactic_powerup(summary: Dict[str, Any], observer: BattlefieldObserver, state: IterState) -> ActionCommand:
    powerups: Dict[str, Dict[str, Any]] = (
        summary.get("logistics", {}).get("powerups") or {}
    )
    if not powerups:
        return tactic_search(summary, observer, state)

    def wrap_angle_deg(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0

    hp_pct = float(summary["self"].get("hp_pct", 100.0))

    my_pos = summary["self"]["pos"]
    mx = float(my_pos["x"])
    my = float(my_pos["y"])
    heading_deg = float(summary["self"]["heading"])

    def is_heal_type(type_name: str) -> bool:
        n = type_name.upper()
        return ("REPAIR" in n) or ("HEAL" in n) or ("MED" in n)

    def score(type_name: str, p: Dict[str, Any]) -> float:
        dist = float(p.get("dist", 1e9))
        val = float(p.get("val", 0))

        heal_boost = 0.0
        if is_heal_type(type_name):
            if hp_pct < 35.0:
                heal_boost = 2000.0
            elif hp_pct < 60.0:
                heal_boost = 1000.0

        efficiency = val / (dist + 1.0)
        far_penalty = 0.002 * dist * dist
        return heal_boost + 10.0 * efficiency - far_penalty


    locked: Optional[Tuple[str, Dict[str, Any]]] = None
    if state.goto_x is not None and state.goto_y is not None:
        gx = float(state.goto_x)
        gy = float(state.goto_y)

        lock_radius = 6.0
        best_match = None
        best_match_d = 1e9
        for tname, p in powerups.items():
            pos = p.get("pos")
            if not pos:
                continue
            px = float(pos["x"])
            py = float(pos["y"])
            d = math.hypot(px - gx, py - gy)
            if d < best_match_d:
                best_match_d = d
                best_match = (tname, p)

        if best_match is not None and best_match_d <= lock_radius:
            locked = best_match

    if locked is None:
        best_type, best = max(powerups.items(), key=lambda kv: score(kv[0], kv[1]))
    else:
        best_type, best = locked

    pos = best["pos"]
    tx = float(pos["x"])
    ty = float(pos["y"])
    dist = float(best.get("dist", math.hypot(tx - mx, ty - my)))

    target_angle = math.degrees(math.atan2(ty - my, tx - mx))
    err = wrap_angle_deg(target_angle - heading_deg)
    aerr = abs(err)

    if dist > 80.0:
        base_speed = 5.0
    elif dist > 30.0:
        base_speed = 4.0
    else:
        base_speed = 3.0

    if aerr > 70.0:
        speed_cmd = 0.0  # obróć się w miejscu
    elif aerr > 40.0:
        speed_cmd = min(base_speed, 2.0)
    else:
        speed_cmd = base_speed

    state.set_goto(x=tx, y=ty, speed=speed_cmd, stop_radius=4.0)

    if state.motion_mode not in (MotionMode.BACKUP, MotionMode.ROTATE):
        state.motion_mode = MotionMode.GOTO

    speed, heading_rot = state.motion_step(
        summary,
        rot_ang=ROT_ANG,
        map_cfg=_map_cfg_from_observer(observer),
        rec_cfg=_rec_cfg_default(),
        enable_border=False,
        enable_obstacle=False,
        enable_stuck=True,
        enable_terrein=False,
    )

    nearest_enemy = summary.get("radar", {}).get("nearest_enemy")
    if nearest_enemy is None:
        barrel_rot = float(ROT_ANG)
        should_fire = False
    else:
        barrel_rot = _clamp_rotation_delta(
            float(summary["tactical"]["rotation_to_target"]),
            BARREL_SPIN_RATE,
        )
        should_fire = bool(summary["tactical"]["can_fire"])

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=float(heading_rot),
        move_speed=float(speed),
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=should_fire,
    )
    

def tactic_reload(summary: Dict[str, Any], observer: BattlefieldObserver, state: IterState) -> ActionCommand:

    nearest = summary["radar"]["nearest_enemy"]

    if nearest:
        barrel_rot = _clamp_rotation_delta(
            float(summary["tactical"]["rotation_to_target"]),
            BARREL_SPIN_RATE,
        )
    else:
        barrel_rot = 30.0

    hp = float(summary["self"]["hp_pct"])

    state.set_forward(speed=-5.0)

    speed, heading_rot = state.motion_step(
        summary,
        rot_ang=ROT_ANG,
        map_cfg=_map_cfg_from_observer(observer),
        rec_cfg=_rec_cfg_default(),
        enable_border=True,
        enable_obstacle=True,
        enable_stuck=True,
        enable_terrein=True if hp < 15.0 else False,
    )
    

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=heading_rot,
        move_speed=speed,
        ammo_to_load=get_best_ammo(observer.my_tank),
        should_fire=False,
    )


def get_action_to_tactics(strategy: StrategyType, observer: BattlefieldObserver, state: IterState) -> ActionCommand:
    """
    Funkcja mapująca wybraną strategię (np. z sieci neuronowej/ANFIS) 
    na konkretne wywołanie taktyki sterującej.
    """
    summary = observer.get_summary()

    tactics_map = {
        StrategyType.ATTACK: tactic_attack,
        StrategyType.FLEE: tactic_flee,
        StrategyType.SAVE: tactic_save,
        StrategyType.SEARCH: tactic_search,
        StrategyType.POWERUP: tactic_powerup,
        #StrategyType.RELOAD: tactic_reload,
    }

    tactic_function = tactics_map.get(strategy, tactic_search)
    return tactic_function(summary, observer, state)