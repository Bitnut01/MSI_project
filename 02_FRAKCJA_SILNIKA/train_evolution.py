import copy
import random
import subprocess
import time
import os

import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

parent_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

agent_dir = os.path.join(os.path.dirname(current_dir), '03_FRAKCJA_AGENTOW', 'agent007')
sys.path.insert(0, agent_dir)

from typing import Dict, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

import json
from typing import Type

import numpy as np
from tqdm import tqdm
from src.genetic import ANFIS_Specimen
from src.strategy import INPUTS_DEFINITION
from backend.engine.game_loop import run_game
from backend.engine import game_loop as game_loop_module
from backend.utils.config import game_config

def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class GeneticTraining:

    BASE_PORT = 8001
    AGENT_LOG_LEVEL = os.getenv("AGENT_LOG_LEVEL", "WARNING")
    AGENT_LOG_ACTIONS = os.getenv("AGENT_LOG_ACTIONS", "0").lower() in {"1", "true", "yes", "on"}
    AGENT_LOG_DIR = os.getenv("AGENT_LOG_DIR", "")
    COMBAT_MODE = os.getenv("TRAIN_COMBAT_MODE", "1").lower() in {"1", "true", "yes", "on"}
    TRAIN_MAP_SEED = os.getenv("TRAIN_MAP_SEED", "open.csv")
    TRAIN_SUDDEN_DEATH_TICK = int(os.getenv("TRAIN_SUDDEN_DEATH_TICK", "460"))
    TRAIN_SUDDEN_DEATH_DAMAGE = int(os.getenv("TRAIN_SUDDEN_DEATH_DAMAGE", "-3"))
    TRAIN_DISABLE_POWERUPS = os.getenv("TRAIN_DISABLE_POWERUPS", "1").lower() in {"1", "true", "yes", "on"}
    TRAIN_FRONT_GAP = float(os.getenv("TRAIN_FRONT_GAP", "24"))
    TRAIN_LANE_SPACING = float(os.getenv("TRAIN_LANE_SPACING", "14"))

    PM = 0.5
    PK = 0.3
    RANK = 0.4
    
    def __init__(self, specimens: list[ANFIS_Specimen], method: str = "ranking", epoch: int = 100 ) -> None:
        self.specimens = specimens
        self.method = method
        self.best = (1e6, None)
        self.epoch = epoch

    def mutate(self) -> None:
        for specimen in self.specimens:
            if random.random() <= self.PM:
                specimen.mutate()

    def create_descendants(self) -> None:
        outGeneration: list[ANFIS_Specimen] = []
        random.shuffle(self.specimens)
        for pair in batched(self.specimens, n=2):
            if random.random() <= self.PK:
                outGeneration.extend(pair[0].create_descendants(pair[1]))
            else:
                outGeneration.extend(pair)
        self.specimens = outGeneration
    
    def validate(self) -> list[float]:
        return [-s.score for s in self.specimens]

    def _build_training_config(self):
        cfg = copy.deepcopy(game_config)
        cfg.game_rules.sudden_death_tick = self.TRAIN_SUDDEN_DEATH_TICK
        cfg.game_rules.sudden_death_damage_per_tick = self.TRAIN_SUDDEN_DEATH_DAMAGE

        if self.TRAIN_DISABLE_POWERUPS:
            cfg.powerup_config.max_powerups_on_map = 0
            cfg.powerup_config.spawn_start_tick = self.TRAIN_SUDDEN_DEATH_TICK + 1

        return cfg

    def _build_close_spawn_points(self, cfg) -> dict[int, list[tuple[float, float]]]:
        team_a_size = game_loop_module.TEAM_A_NBR
        team_b_size = game_loop_module.TEAM_B_NBR
        map_width = float(cfg.map_config.width)
        map_height = float(cfg.map_config.height)

        center_x_env = os.getenv("TRAIN_CENTER_X")
        center_y_env = os.getenv("TRAIN_CENTER_Y")
        center_x = float(center_x_env) if center_x_env is not None else map_width / 2.0
        center_y = float(center_y_env) if center_y_env is not None else map_height / 2.0

        margin = 12.0
        front_gap_half = self.TRAIN_FRONT_GAP / 2.0
        max_team_size = max(team_a_size, team_b_size)
        lane_mid = (max_team_size - 1) / 2.0

        def clamp(value: float, low: float, high: float) -> float:
            return max(low, min(value, high))

        team_a_points = []
        for idx in range(team_a_size):
            y = center_y + (idx - lane_mid) * self.TRAIN_LANE_SPACING
            team_a_points.append(
                (
                    clamp(center_x - front_gap_half, margin, map_width - margin),
                    clamp(y, margin, map_height - margin),
                )
            )

        team_b_points = []
        for idx in range(team_b_size):
            y = center_y + (idx - lane_mid) * self.TRAIN_LANE_SPACING
            team_b_points.append(
                (
                    clamp(center_x + front_gap_half, margin, map_width - margin),
                    clamp(y, margin, map_height - margin),
                )
            )

        return {1: team_a_points, 2: team_b_points}

    def _selection_ranking(self) -> None:
        if not self.specimens:
            return

        population_size = len(self.specimens)
        elite_count = max(1, int(population_size * self.RANK))

        # Higher score is better.
        ranked = sorted(self.specimens, key=lambda specimen: specimen.score, reverse=True)
        elites = [copy.deepcopy(specimen) for specimen in ranked[:elite_count]]
        survivors = [copy.deepcopy(specimen) for specimen in ranked[:-elite_count]]

        selected = elites + survivors
        self.specimens = selected[:population_size]

    def _selection_roulette(self) -> None:
        score = self.validate()
        worst_score = max(score) + 1
        self.specimens = random.choices(self.specimens, weights = [(worst_score - s)*10 for s in score], k=len(self.specimens))

    def selection(self) -> None:
        match self.method:
            case "ranking":
                self._selection_ranking()
                pass
            case "roulette":
                self._selection_roulette()
                pass
            case _:
                raise ValueError("Unknown selection method")

    def fit(self) -> None:
        pbar = tqdm(range(self.epoch), desc=f"")
        for epoch_idx in pbar:
            self.create_descendants()
            self.mutate()
            
            self.run_generation()

            self.selection()
            epoch_best = copy.deepcopy(min(zip(self.validate(), self.specimens), key=lambda pair: pair[0]))
            epoch_best[1].save_to_file(f"best_s{-epoch_best[0]}_e{epoch_idx}.json")
            if epoch_best[0] < self.best[0]:
                self.best = epoch_best

    def run_generation(self):
        b = 0
        for specimen_batch in batched(self.specimens, n = 10):
            agent_processes = []
            for i, specimen in enumerate(specimen_batch):
                port = self.BASE_PORT + i
                gene_path = f"genes_bot_{port + b*10}.json"
                specimen.save_to_file(gene_path)

                proc = subprocess.Popen([
                    sys.executable, os.path.join(agent_dir, "run_agent.py"), 
                    "--port", str(port),
                    "--genes", gene_path,
                    "--train",
                    "--log-level", self.AGENT_LOG_LEVEL,
                    *(
                        ["--log-file", os.path.join(self.AGENT_LOG_DIR, f"agent_{port}.log")]
                        if self.AGENT_LOG_DIR
                        else []
                    ),
                    *(["--log-actions"] if self.AGENT_LOG_ACTIONS else []),
                ])
                agent_processes.append(proc)

            # Czas na wstanie FastAPI
            time.sleep(1)

            # 2. Uruchom silnik gry (Headless)
            try:
                if self.COMBAT_MODE:
                    cfg = self._build_training_config()
                    spawn_points = self._build_close_spawn_points(cfg)
                    game_loop = game_loop_module.GameLoop(
                        config=cfg,
                        headless=False,
                        spawn_points=spawn_points,
                    )
                    try:
                        if game_loop.initialize_game(map_seed=self.TRAIN_MAP_SEED):
                            results = game_loop.run_game_loop()
                        else:
                            results = {"success": False, "error": "Initialization failed"}
                    finally:
                        game_loop.cleanup_game()
                else:
                    results = run_game(headless=True)
                print(results)
            finally:
                for p in agent_processes:
                    p.terminate()
            
            b += 1
        
        
        for i in range(len(self.specimens)):
            self.specimens[i] = ANFIS_Specimen.load_from_file(f"genes_bot_{self.BASE_PORT + i}.json")
        self._print_strategy_counts(len(self.specimens))
            

    def _print_strategy_counts(self, specimen_count: int) -> None:
        totals: dict[str, int] = {}
        for i in range(specimen_count):
            port = self.BASE_PORT + i
            stats_path = f"strategy_counts_Agent007_{port}.json"
            if not os.path.exists(stats_path):
                continue
            try:
                with open(stats_path, "r", encoding="utf-8") as handle:
                    counts = json.load(handle)
            except Exception:
                continue

            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + int(value)

        if totals:
            ordered = sorted(totals.items(), key=lambda item: item[0])
            summary = ", ".join(f"{k}={v}" for k, v in ordered)
            print(f"Strategy counts (epoch): {summary}")

    def get_best(self) -> tuple[float, Type[ANFIS_Specimen]]:
        return self.best
    

def main():
    specimens = [
        # ANFIS_Specimen.generate_random(INPUTS_DEFINITION) for _ in range(100)
        ANFIS_Specimen.load_from_file(f"genes_bot_{8001 + i}.json") for i in range(100)
    ]
    trainer = GeneticTraining(specimens, "ranking", epoch=4)
    trainer.fit()

if __name__ == "__main__":
    sys.exit(main())