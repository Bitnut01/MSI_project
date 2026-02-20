import random
import numpy as np
import logging

from pydantic import BaseModel
from typing import Dict, Any, Optional, Tuple

from .observer import BattlefieldObserver
from .strategy import StrategyType, StrategyModel, INPUTS_DEFINITION

from .genetic import ANFIS_Specimen

from .tactics import *

# ============================================================================
# ACTION COMMAND MODEL
# ============================================================================

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

# ============================================================================
# AGENT LOGIC
# ============================================================================

class Agent007:
    """
    Agent with more structured, stateful behavior for testing purposes.
    Drives in one direction for a while, then changes.
    Scans with its turret.
    """
    
    def __init__(self, name: str = "Bot007", specimen: ANFIS_Specimen = None, training: bool = False):
        self.name = name
        self.is_destroyed = False
        self.logger = logging.getLogger("agent007")
        self.logger.info("[%s] Agent initialized", self.name)

        # RANDOM CODE ======================================================
        # State for movement
        self.move_timer = 0
        self.current_move_speed = 0.0

        # State for hull rotation
        self.heading_timer = 0
        self.current_heading_rotation = 0.0

        # State for barrel scanning
        self.barrel_scan_direction = 1.0  # 1.0 for right, -1.0 for left
        self.barrel_rotation_speed = 15.0

        # State for aiming before shooting
        self.aim_timer = 0  # Ticks to wait before firing
        # ==================================================================

        self.observer = BattlefieldObserver(training_mode=training)
        self.score = 0
        self.specimen = None
        self.training = training
        self.strategy_selector = StrategyModel(INPUTS_DEFINITION)
        self.strategy_ticks = {strategy: 0 for strategy in StrategyType}
        self.strategy_switches = 0
        self.last_strategy = None
        self.total_decisions = 0
        self.shots_attempted = 0
        self.log_actions = False
        if specimen:
            self.load_specimen(specimen)
            
    def load_specimen(self, specimen: ANFIS_Specimen):
        self.specimen = specimen
        self.strategy_selector.set_params_from_genes(specimen)

    def set_training_mode(self, enabled: bool) -> None:
        self.training = enabled
        self.observer.set_training_mode(enabled)

    def _prepare_inputs(self, summary: dict) -> np.ndarray:
        """Mapuje dane z Observera na zakres [0, 1] dla ANFIS."""
        enemy = summary.get("radar", {}).get("nearest_enemy")
        enemy_dist = enemy.get("dist") if enemy else None

        powerups = summary.get("logistics", {}).get("powerups", {})
        nearest_powerup_dist = None
        for powerup in powerups.values():
            dist = powerup.get("dist")
            if dist is None:
                continue
            if nearest_powerup_dist is None or dist < nearest_powerup_dist:
                nearest_powerup_dist = dist

        feature_values = {
            "my_hp": summary.get("self", {}).get("hp_pct", 100.0) / 100.0,
            "enemy_dist": (enemy_dist / 300.0) if enemy_dist is not None else 1.0,
            "reload_status": summary.get("self", {}).get("reload_ticks", 0.0) / 10.0,
            "aim_error": abs(summary.get("tactical", {}).get("rotation_to_target", 0.0)) / 180.0,
            "powerup": (
                nearest_powerup_dist / 300.0 if nearest_powerup_dist is not None else 1.0
            ),
            "can_fire": 1.0 if summary.get("tactical", {}).get("can_fire", False) else 0.0,
            # na bezwzględnej wartości obrażeń
            "terrain_risk": abs(float(summary.get("self", {}).get("terrain_damage", 0.0) or 0.0)) / 5.0,
        }

        ordered_features = []
        for fuzzy_input in INPUTS_DEFINITION:
            feature_name = getattr(fuzzy_input, "name", "")
            value = feature_values.get(feature_name, 0.5)
            ordered_features.append(float(np.clip(value, 0.0, 1.0)))

        return np.array(ordered_features, dtype=float)

    def _nearest_powerup_distance(self, summary: Dict[str, Any]) -> Optional[float]:
        powerups = summary.get("logistics", {}).get("powerups", {})
        distances = [
            pu.get("dist")
            for pu in powerups.values()
            if isinstance(pu, dict) and pu.get("dist") is not None
        ]
        if not distances:
            return None
        return float(min(distances))

    def _map_prediction_to_strategy(self, raw_prediction: float) -> StrategyType:
        # Use nearest strategy bucket instead of floor to reduce low-index bias.
        clipped = float(np.clip(raw_prediction, 0.0, len(StrategyType) - 1))
        strategy_idx = int(np.floor(clipped + 0.5))
        return StrategyType(strategy_idx)

    def _apply_safety_fallback(
        self, summary: Dict[str, Any], model_strategy: StrategyType
    ) -> Tuple[StrategyType, Optional[str]]:
        nearest_enemy = summary.get("radar", {}).get("nearest_enemy")
        enemy_dist = (
            float(nearest_enemy.get("dist"))
            if nearest_enemy and nearest_enemy.get("dist") is not None
            else None
        )
        hp_pct = float(summary.get("self", {}).get("hp_pct", 100.0) or 100.0)
        reload_ticks = float(summary.get("self", {}).get("reload_ticks", 0.0) or 0.0)
        can_fire = bool(summary.get("tactical", {}).get("can_fire", False))
        terrain_damage = float(summary.get("self", {}).get("terrain_damage", 0.0) or 0.0)
        nearest_powerup_dist = self._nearest_powerup_distance(summary)

        # Hard safety rules: stay alive first.
        if hp_pct <= 25.0 and enemy_dist is not None and enemy_dist <= 220.0:
            return StrategyType.FLEE, "critical_hp"

        if terrain_damage <= -3.0 and enemy_dist is not None and enemy_dist <= 180.0:
            return StrategyType.FLEE, "deadly_terrain"

        # Avoid standing still under threat when weapon is reloading.
        if reload_ticks > 0.0 and enemy_dist is not None and enemy_dist <= 180.0:
            return StrategyType.RELOAD, "reload_under_threat"

        # If we have a clear close shot, force aggression.
        if can_fire and enemy_dist is not None and enemy_dist <= 200.0 and hp_pct > 20.0:
            return StrategyType.ATTACK, "attack_window"

        # If no enemy is visible, do not sit in defensive mode.
        if enemy_dist is None:
            if nearest_powerup_dist is not None and nearest_powerup_dist <= 90.0 and hp_pct < 80.0:
                return StrategyType.POWERUP, "safe_powerup_pickup"
            return StrategyType.SEARCH, "no_enemy_visible"

        # Nearby useful pickup while not in immediate danger.
        if (
            nearest_powerup_dist is not None
            and nearest_powerup_dist <= 100.0
            and (hp_pct < 60.0 or enemy_dist > 260.0)
        ):
            return StrategyType.POWERUP, "near_powerup"

        return model_strategy, None

    def decide_strategy(
        self, summary: Dict[str, Any]
    ) -> Tuple[StrategyType, float, StrategyType, Optional[str]]:
        """Returns final strategy, raw model output, model strategy and fallback reason."""
        input_vector = self._prepare_inputs(summary)
        prediction = self.strategy_selector.get_result(input_vector)
        raw_prediction = (
            prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
        )
        raw_prediction = float(raw_prediction)

        model_strategy = self._map_prediction_to_strategy(raw_prediction)
        final_strategy, fallback_reason = self._apply_safety_fallback(
            summary, model_strategy
        )
        return final_strategy, raw_prediction, model_strategy, fallback_reason

    def _update_strategy_metrics(self, strategy: StrategyType) -> None:
        self.total_decisions += 1
        self.strategy_ticks[strategy] += 1
        if self.last_strategy is not None and self.last_strategy != strategy:
            self.strategy_switches += 1
        self.last_strategy = strategy

    def get_action(
        self, 
        current_tick: int, 
        my_tank_status: Dict[str, Any], 
        sensor_data: Dict[str, Any], 
        enemies_remaining: int
    ) -> ActionCommand:
        
        # NEW CODE =========================================================
        self.observer.update(my_tank_status, sensor_data, enemies_remaining)
        summary = self.observer.get_summary()
        
        current_strategy, raw_prediction, model_strategy, fallback_reason = self.decide_strategy(summary)
        self._update_strategy_metrics(current_strategy)
        # ==================================================================


        action = get_action_to_tactics(current_strategy, self.observer)
        if action.should_fire:
            self.shots_attempted += 1

        if self.log_actions:
            action_dump = (
                action.model_dump()
                if hasattr(action, "model_dump")
                else action.dict()
            )
            self.logger.info(
                "[%s] tick=%s strategy=%s model=%s raw=%.3f fallback=%s enemies=%s action=%s",
                self.name,
                current_tick,
                current_strategy.name,
                model_strategy.name,
                raw_prediction,
                fallback_reason or "-",
                enemies_remaining,
                action_dump,
            )

        return action
        
        # should_fire = False
        # heading_rotation = 0.0
        # barrel_rotation = 0.0
        
        # if self.aim_timer > 0:
        #     # --- AIMING PHASE ---
        #     self.aim_timer -= 1
            
        #     # Stop all rotation while aiming
        #     heading_rotation = 0.0
        #     barrel_rotation = 0.0
            
        #     # Fire on the last tick of aiming
        #     if self.aim_timer == 0:
        #         should_fire = True
        # else:
        #     # --- NORMAL OPERATION PHASE ---

        #     # --- Hull Rotation Logic ---
        #     self.heading_timer -= 1
        #     if self.heading_timer <= 0:
        #         self.current_heading_rotation = random.choice([-15.0, 0, 15.0])
        #         self.heading_timer = random.randint(30, 90)
        #     heading_rotation = self.current_heading_rotation

        #     # --- Barrel Scanning Logic ---
        #     barrel_angle = my_tank_status.get("barrel_angle", 0.0)
        #     if barrel_angle > 45.0:
        #         self.barrel_scan_direction = -1.0  # Scan left
        #     elif barrel_angle < -45.0:
        #         self.barrel_scan_direction = 1.0  # Scan right
        #     barrel_rotation = self.barrel_rotation_speed * self.barrel_scan_direction

        #     # --- Shooting Decision ---
        #     # Decide if we should start aiming
        #     wants_to_shoot = random.random() < 0.3
        #     if wants_to_shoot:
        #         self.aim_timer = 10  # Start aiming for 10 ticks

        # # --- Movement Logic (independent of aiming) ---
        # self.move_timer -= 1
        # if self.move_timer <= 0:
        #     self.current_move_speed = random.choice([30.0, 30.0, 0.0, -10.0])
        #     self.move_timer = random.randint(1, 10)
            
        # ammo_data = my_tank_status.get("ammo", {})
        # best_ammo_type = None

        # if ammo_data:
        #     # Znajduje klucz (nazwę amunicji), który ma największą wartość w polu 'count'
        #     best_ammo_type = max(ammo_data, key=lambda k: ammo_data[k].get("count", 0))

        # return ActionCommand(
        #     barrel_rotation_angle  = barrel_rotation,
        #     heading_rotation_angle = heading_rotation,
        #     move_speed             = self.current_move_speed,
        #     ammo_to_load           = best_ammo_type,
        #     should_fire            = should_fire and summary["tactical"]["can_fire"]
        # )

    def destroy(self):
        """Called when tank is destroyed."""
        self.is_destroyed = True
        self.logger.info("[%s] Tank destroyed!", self.name)
    
    def end(self, damage_dealt: float, tanks_killed: int):
        """Called when game ends."""
        self.logger.info("[%s] Game ended!", self.name)
        self.logger.info("[%s] Damage dealt: %s", self.name, damage_dealt)
        self.logger.info("[%s] Tanks killed: %s", self.name, tanks_killed)
        if self.training and self.specimen:
            self._score_genotype(damage_dealt, tanks_killed)
            
    def _score_genotype(self, damage_dealt, tanks_killed):
        strategy_ratios = {strategy: 0.0 for strategy in StrategyType}
        switch_ratio = 0.0

        if self.total_decisions > 0:
            usage = []
            for strategy in StrategyType:
                ratio = self.strategy_ticks[strategy] / self.total_decisions
                strategy_ratios[strategy] = ratio
                usage.append(f"{strategy.name}={ratio:.2%}")
            if self.total_decisions > 1:
                switch_ratio = self.strategy_switches / (self.total_decisions - 1)
            self.logger.info("[%s] Strategy usage: %s", self.name, ", ".join(usage))
            self.logger.info(
                "[%s] Strategy switches: %s (%.2f%%)",
                self.name,
                self.strategy_switches,
                switch_ratio * 100.0,
            )

        damage = float(damage_dealt)
        kills = float(tanks_killed)
        hp_left = float(self.observer.my_tank.get("hp", 0.0))

        # Combat-first reward: damage and kills dominate survival reward.
        score = 0.0
        score += damage * 1.15
        score += kills * 85.0
        score += hp_left * 0.10
        score += 8.0 if not self.is_destroyed else -18.0

        # Encourage actual firing behavior.
        score += min(self.shots_attempted, 80) * 0.20
        if self.shots_attempted == 0:
            score -= 15.0

        # Encourage active combat-oriented strategies.
        score += strategy_ratios[StrategyType.ATTACK] * 26.0
        score += strategy_ratios[StrategyType.RELOAD] * 8.0
        score += strategy_ratios[StrategyType.FLEE] * 4.0

        # Penalize overly passive policies.
        score -= max(0.0, strategy_ratios[StrategyType.SAVE] - 0.35) * 38.0
        score -= max(0.0, strategy_ratios[StrategyType.SEARCH] - 0.30) * 34.0

        # Reward moderate adaptation, penalize twitchy switching.
        target_switch_ratio = 0.15
        score += max(0.0, 1.0 - abs(switch_ratio - target_switch_ratio) / target_switch_ratio) * 8.0
        if switch_ratio > 0.50:
            score -= (switch_ratio - 0.50) * 35.0

        # Hard penalties for non-combat outcomes.
        if damage < 5.0 and kills == 0:
            score -= 25.0
        if damage == 0.0 and kills == 0:
            score -= 35.0

        self.specimen.score = float(score)
        self.specimen.save_to_file()
        self.logger.info(
            "[%s] Final specimen score: %.3f (damage=%.1f kills=%s hp=%.1f shots=%s)",
            self.name,
            score,
            damage,
            tanks_killed,
            hp_left,
            self.shots_attempted,
        )

