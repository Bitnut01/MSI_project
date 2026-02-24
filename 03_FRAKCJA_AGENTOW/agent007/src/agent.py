import random
import numpy as np

from pydantic import BaseModel
from typing import Dict, Any

from .observer import BattlefieldObserver
from .strategy import StrategyType, StrategyModel, INPUTS_DEFINITION

from .genetic import ANFIS_Specimen

from .tactics import *
from .state import *

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
        print(f"[{self.name}] Agent initialized")

        # RANDOM CODE ======================================================
        # # State for movement
        # self.move_timer = 0
        # self.current_move_speed = 0.0

        # # State for hull rotation
        # self.heading_timer = 0
        # self.current_heading_rotation = 0.0

        # # State for barrel scanning
        # self.barrel_scan_direction = 1.0  # 1.0 for right, -1.0 for left
        # self.barrel_rotation_speed = 15.0

        # # State for aiming before shooting
        # self.aim_timer = 0  # Ticks to wait before firing
        # ==================================================================

        # self.tactic_state = {
        #     "on_bad_terrain": False,
        #     "escape_timer": 10,
        #     "search_turn_timer": 300,
        #     "current_path": [],
        #     "last_pos": {"x": None, "y": None},
        #     "stucked": False,
        #     "stucked_timer": 2,
        #     "rotation_timer": 2,
        #     "stucked_timer_2": 50,
        #     "rotation_dir": 1, # 1 = left, -1 = right
        #     "is_rotating": False,
        #     "last_heading": 0.0,
        #     "boarder_time": 10,
        # }

        self.tactic_state = IterState()

        self.observer = BattlefieldObserver()
        self.score = 0
        self.specimen = None
        self.training = training
        self.strategy_selector = StrategyModel(INPUTS_DEFINITION)
        if specimen:
            self.load_specimen(specimen)
    
    def set_training_mode(self, enabled: bool) -> None:
        self.training = enabled
        self.observer.set_training_mode(enabled)
            
    def load_specimen(self, specimen: ANFIS_Specimen):
        self.specimen = specimen
        self.strategy_selector.set_params_from_genes(specimen)

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


    # def _prepare_inputs(self, summary: dict) -> np.ndarray:
    #     """Mapuje dane z Observera na zakres [0, 1] dla ANFIS."""
    #     # 1. HP (0-100 -> 0-1)
    #     hp = summary["self"]["hp_pct"] / 100.0

    #     # 2. Dystans do wroga (0-800 -> 0-1)
    #     enemy = summary["radar"]["nearest_enemy"]
    #     e_dist = min(enemy["dist"] / 800.0, 1.0) if enemy else 1.0
        
    #     # 3. Status przeładowania (0-1)
    #     reload = min(summary["self"]["reload_ticks"] / 60.0, 1.0)
        
    #     return np.array([hp, e_dist, reload])

    def decide_strategy(self, summary) -> StrategyType:
        """Główna metoda wyboru strategii."""
        input_vector = self._prepare_inputs(summary)
        prediction = self.strategy_selector.get_result(input_vector)
        val = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

        return StrategyType(int(np.clip(np.floor(val), 0, 5)))

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
        
        current_strategy = self.decide_strategy(summary)
        #print(summary)
        # ==================================================================
        # powerup = summary["logistics"]["powerups"]

        action = get_action_to_tactics(current_strategy, self.observer, self.tactic_state)

        return action

    def destroy(self):
        """Called when tank is destroyed."""
        self.is_destroyed = True
        print(f"[{self.name}] Tank destroyed!")
    
    def end(self, damage_dealt: float, tanks_killed: int):
        """Called when game ends."""
        print(f"[{self.name}] Game ended!")
        print(f"[{self.name}] Damage dealt: {damage_dealt}")
        print(f"[{self.name}] Tanks killed: {tanks_killed}")
        if self.training and self.specimen:
            self._score_genotype(damage_dealt, tanks_killed)
            
    def _score_genotype(self, damage_dealt, tanks_killed):
        score = damage_dealt/10
        score += tanks_killed*10
        score -= self.is_destroyed*50
        score +=  self.observer.my_tank["hp"]
        self.specimen.score = score
        self.specimen.save_to_file()