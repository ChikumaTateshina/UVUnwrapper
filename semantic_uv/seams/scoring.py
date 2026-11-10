"""シーム候補スコア。値が大きいほど切断候補として有力。"""
from __future__ import annotations

import math

import numpy as np

from ..analysis.curvature import edge_dihedral, vertex_angle_defect
from ..analysis.materials import material_boundaries
from ..analysis.shape_features import edge_lengths_world
from ..analysis.topology import Topology
from ..core.models import EdgeFeatures, ObjectData
from .constraints import ResolvedConstraints

W_DIHEDRAL = 1.0
W_MATERIAL = 0.45
W_CONCAVE = 0.15
W_COLOR = 0.2
W_CURVATURE = 0.1


def compute_edge_features(obj: ObjectData, topo: Topology,
                          rc: ResolvedConstraints) -> EdgeFeatures:
    dihedral, concavity = edge_dihedral(topo)
    mat_b, color = material_boundaries(topo)
    lengths = edge_lengths_world(topo)
    defect = np.abs(vertex_angle_defect(topo))
    E = obj.n_edges
    score = np.zeros(E)
    d_norm = np.nan_to_num(dihedral / math.pi, nan=0.0)
    score += W_DIHEDRAL * d_norm
    score += W_MATERIAL * mat_b.astype(float)
    # 凹エッジ (くびれ) は目立ちにくいとみなす近似
    score += W_CONCAVE * ((np.nan_to_num(concavity, nan=0.0) < 0) & (d_norm > 0.05)).astype(float)
    score += W_COLOR * np.clip(np.nan_to_num(color, nan=0.0), 0, 1)
    if E:
        ev = obj.edges
        curv = np.clip((defect[ev[:, 0]] + defect[ev[:, 1]]) / math.pi, 0, 1)
        score += W_CURVATURE * curv
    score += 0.5 * rc.soft_bias
    # 境界エッジは常に島の境界
    faces = [tuple(fs) for fs in topo.edge_faces]
    return EdgeFeatures(
        object_key=obj.key, faces=faces, length=lengths, dihedral=dihedral,
        concavity=concavity, material_boundary=mat_b, color_delta=color,
        existing_seam=obj.seams.copy(), constraint=obj.edge_constraint.copy(),
        score=score)
