from threestudio.utils.typing import *


class Mesh:
    def __init__(
        self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"]
    ) -> None:
        self.v_pos = v_pos
        self.t_pos_idx = t_pos_idx
