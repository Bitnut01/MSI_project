import random
import numpy as np

from pydantic import BaseModel
from typing import Dict, Any

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
        print(f"[{self.name}] Agent initialized")

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

        self.observer = BattlefieldObserver()
        self.score = 0
        self.specimen = None
        self.training = training
        self.strategy_selector = StrategyModel(INPUTS_DEFINITION)
        if specimen:
            self.load_specimen(specimen)
            
    def load_specimen(self, specimen: ANFIS_Specimen):
        self.specimen = specimen
        self.strategy_selector.set_params_from_genes(specimen)

    def _prepare_inputs(self, summary: dict) -> np.ndarray:
        """Mapuje dane z Observera na zakres [0, 1] dla ANFIS."""
        # 1. HP (0-100 -> 0-1)
        hp = summary["self"]["hp_pct"] / 100.0

        # 2. Dystans do wroga (0-800 -> 0-1)
        enemy = summary["radar"]["nearest_enemy"]
        e_dist = min(enemy["dist"] / 800.0, 1.0) if enemy else 1.0
        
        # 3. Status przeładowania (0-1)
        reload = min(summary["self"]["reload_ticks"] / 60.0, 1.0)
        
        return np.array([hp, e_dist, reload])

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
        print(current_strategy.name)
        # ==================================================================

        current_strategy = StrategyType.SEARCH


        action = get_action_to_tactics(current_strategy, self.observer)

        print(action)

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