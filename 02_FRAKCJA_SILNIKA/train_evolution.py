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

def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class GeneticTraining:

    BASE_PORT = 8001

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

    def _selection_ranking(self) -> None:
        sort = [x for _, x in sorted(zip(self.validate(), self.specimens), key=lambda pair: pair[0], reverse=True)]
        dropIndex = int(len(self.specimens)*self.RANK)
        best = [s for s in sort[:dropIndex]]
        best.extend(sort[:-dropIndex])
        self.specimens = best

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
            epoch_best[1].save_to_file(f"best_s{epoch_best[0]}_e{epoch_idx}.json")
            if epoch_best[0] < self.best[0]:
                self.best = epoch_best

    def run_generation(self):
        agent_processes = []
        
        for i, specimen in enumerate(self.specimens):
            port = self.BASE_PORT + i
            gene_path = f"genes_bot_{port}.json"
            specimen.save_to_file(gene_path)

            proc = subprocess.Popen([
                sys.executable, os.path.join(agent_dir, "run_agent.py"), 
                "--port", str(port), 
                "--name", f"Bot_{port}",
                "--genes", gene_path
            ])
            agent_processes.append(proc)

         # Czas na wstanie FastAPI
        time.sleep(5)

        # 2. Uruchom silnik gry (Headless)
        try:
            results = run_game(map_seed="1234", headless=True)
            print(results)
        finally:
            for p in agent_processes:
                p.terminate()
            for i in range(len(self.specimens)):
                self.specimens[i] = ANFIS_Specimen.load_from_file(f"genes_bot_{self.BASE_PORT + i}.json")

    def get_best(self) -> tuple[float, Type[ANFIS_Specimen]]:
        return self.best
    

def main():
    specimens = [
        ANFIS_Specimen.generate_random(INPUTS_DEFINITION) for _ in range(10)
    ]
    trainer = GeneticTraining(specimens, "ranking", epoch=3)
    trainer.fit()

if __name__ == "__main__":
    sys.exit(main())