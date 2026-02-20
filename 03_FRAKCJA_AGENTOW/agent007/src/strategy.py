from enum import IntEnum

import numpy as np
from .ANFIS.params import FuzzyInputVariable_3Trapezoids, FuzzyInputVariable_List_Trapezoids
from .ANFIS.ANFIS import ANFIS
from .genetic import ANFIS_Specimen

class StrategyType(IntEnum):
    SAVE = 0       # Defensywa, szukanie osłon
    FLEE = 1       # Ucieczka od najbliższego wroga
    ATTACK = 2     # Agresywne dążenie do kontaktu i strzału
    SEARCH = 3     # Patrolowanie, szukanie wrogów
    POWERUP = 4    # Ignorowanie walki na rzecz wzmocnień
    RELOAD = 5     # Wycofanie się w celu przeładowania

INPUTS_DEFINITION = [
    FuzzyInputVariable_3Trapezoids(0.5, 0.2, 0.1, 0.1, "my_hp", ["low", "id", "high"]),
    FuzzyInputVariable_3Trapezoids(0.5, 0.2, 0.1, 0.1, "enemy_dist", ["near", "medium", "far"]),
    FuzzyInputVariable_3Trapezoids(0.5, 0.2, 0.1, 0.1, "reload_status", ["ready", "busy", "empty"])
    # FuzzyInputVariable_List_Trapezoids(
    #     [[0.2, 0.1, 0.1, 0.1], [0.8, 0.1, 0.1, 0.1]], 
    #     "powerup", ["blisko", "dalego"]
    # ),
]

class StrategyModel:
    def __init__(self, fuzzy_inputs, output_range: tuple[float, float] = (0.0, 6.0)):
        """
        :param fuzzy_inputs: Lista gotowych obiektów FuzzyInputVariable_List_Trapezoids
        :param output_range: Zakres wartości wyjściowych (domyślnie 0-6)
        """

        dummy_x = np.zeros((len(fuzzy_inputs), 1))
        dummy_y = np.zeros(1)
        
        self.model = ANFIS(fuzzy_inputs, dummy_x, dummy_y)
        self.inputs = fuzzy_inputs
        self.output_range = output_range

        # Zapamiętujemy rozmiary struktur, aby wiedzieć jak ciąć tablicę flatten()
        self._premises_shape = np.shape(self.model.premises)
        self._op_shape = np.shape(self.model.op)
        self._tsk_shape = np.shape(self.model.tsk)
        self._split_indices = [
            np.prod(self._premises_shape), 
            np.prod(self._premises_shape) + np.prod(self._op_shape)
        ]

    def set_params_from_genes(self, specimen: ANFIS_Specimen):
        """Wstrzykuje geny osobnika do struktur modelu ANFIS"""
        p_flat, o_flat, t_flat = np.split(specimen.flatten(), self._split_indices)
        self.model.set_premises_parameters(p_flat.reshape(self._premises_shape))
        self.model.op = o_flat.reshape(self._op_shape)
        self.model.tsk = t_flat.reshape(self._tsk_shape)

    def get_result(self, inputs_vector: np.ndarray) -> float:
        """Pobiera wynik dla pojedynczego stanu gry [0, 1]"""
        self.model.training_data = inputs_vector.reshape(-1, 1)
        res = self.model.get_results()
        return float(res[0])