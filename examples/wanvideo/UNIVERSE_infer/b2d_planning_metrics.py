from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw


def _fill_poly_mask(mask: np.ndarray, poly: np.ndarray, value: int = 1) -> None:
    if mask.ndim != 2:
        return
    poly = np.asarray(poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return
    pil_img = Image.fromarray(mask, mode="L")
    draw = ImageDraw.Draw(pil_img)
    draw.polygon([(float(x), float(y)) for x, y in poly[:, :2]], fill=int(value))
    mask[...] = np.asarray(pil_img, dtype=np.uint8)


class B2DPlanningMetricLite:
    """
    Lightweight STP3-style planning metric implementation used by Bench2Drive eval.
    Kept local to avoid runtime dependency on external metric packages.
    """

    def __init__(self) -> None:
        self.X_BOUND = np.array([-50.0, 50.0, 0.5], dtype=np.float32)
        self.Y_BOUND = np.array([-50.0, 50.0, 0.5], dtype=np.float32)
        self.bev_resolution = np.array([self.X_BOUND[2], self.Y_BOUND[2]], dtype=np.float32)
        self.bev_start_position = np.array(
            [self.X_BOUND[0] + self.X_BOUND[2] / 2.0, self.Y_BOUND[0] + self.Y_BOUND[2] / 2.0],
            dtype=np.float32,
        )
        self.bev_dimension = np.array(
            [
                int((self.X_BOUND[1] - self.X_BOUND[0]) / self.X_BOUND[2]),
                int((self.Y_BOUND[1] - self.Y_BOUND[0]) / self.Y_BOUND[2]),
            ],
            dtype=np.int32,
        )
        self.dx = np.array([self.X_BOUND[2], self.Y_BOUND[2]], dtype=np.float32)
        self.bx = np.array(
            [self.X_BOUND[0] + self.X_BOUND[2] / 2.0, self.Y_BOUND[0] + self.Y_BOUND[2] / 2.0],
            dtype=np.float32,
        )
        self.ego_width = 1.85
        self.ego_length = 4.084
        self.vehicle_indices = {0, 1, 2}
        self.human_indices = {3, 7}
        self._ego_rc = self._build_ego_footprint_rc()

    @property
    def bev_h(self) -> int:
        return int(self.bev_dimension[0])

    @property
    def bev_w(self) -> int:
        return int(self.bev_dimension[1])

    def _build_ego_footprint_rc(self) -> np.ndarray:
        pts = np.array(
            [
                [-self.ego_length / 2.0 + 0.5, self.ego_width / 2.0],
                [self.ego_length / 2.0 + 0.5, self.ego_width / 2.0],
                [self.ego_length / 2.0 + 0.5, -self.ego_width / 2.0],
                [-self.ego_length / 2.0 + 0.5, -self.ego_width / 2.0],
            ],
            dtype=np.float32,
        )
        pts = (pts - self.bx[None, :]) / self.dx[None, :]
        pts[:, [0, 1]] = pts[:, [1, 0]]
        poly = np.round(np.stack([pts[:, 0], pts[:, 1]], axis=-1)).astype(np.int32)
        mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        _fill_poly_mask(mask, poly, value=1)
        rr, cc = np.where(mask > 0)
        if rr.size == 0:
            rr = np.array([0], dtype=np.int32)
            cc = np.array([0], dtype=np.int32)
        return np.stack([rr, cc], axis=-1).astype(np.float32)

    def _agent_poly_region(self, x_a: float, y_a: float, yaw_a: float, length: float, width: float) -> np.ndarray:
        lidar2cv_rot = np.array([[1, 0], [0, -1]], dtype=np.float32)
        trans_a = np.array([[x_a], [y_a]], dtype=np.float32)
        rot_mat_a = np.array(
            [[np.cos(yaw_a), -np.sin(yaw_a)], [np.sin(yaw_a), np.cos(yaw_a)]],
            dtype=np.float32,
        )
        agent_corner = np.array(
            [
                [length / 2.0, -length / 2.0, -length / 2.0, length / 2.0],
                [width / 2.0, width / 2.0, -width / 2.0, -width / 2.0],
            ],
            dtype=np.float32,
        )
        agent_corner_lidar = rot_mat_a @ agent_corner + trans_a
        agent_corner_cv = (
            (lidar2cv_rot @ agent_corner_lidar)
            - self.bev_start_position[:2, None]
            + self.bev_resolution[:2, None] / 2.0
        ).T / self.bev_resolution[:2]
        return np.round(agent_corner_cv).astype(np.int32)

    def build_occupancy(self, gt_boxes: np.ndarray, gt_attr: np.ndarray, n_future: int) -> np.ndarray:
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        gt_attr = np.asarray(gt_attr, dtype=np.float32)
        n_future = max(0, int(n_future))
        if n_future <= 0:
            return np.zeros((0, self.bev_h, self.bev_w), dtype=np.uint8)
        if gt_boxes.ndim != 2 or gt_attr.ndim != 2 or gt_boxes.shape[0] <= 0 or gt_attr.shape[0] <= 0:
            return np.zeros((n_future, self.bev_h, self.bev_w), dtype=np.uint8)

        n_agents = min(int(gt_boxes.shape[0]), int(gt_attr.shape[0]))
        gt_boxes = gt_boxes[:n_agents]
        gt_attr = gt_attr[:n_agents]

        attr_dim = int(gt_attr.shape[1])
        t_attr = (attr_dim - 10) // 4
        if t_attr <= 0:
            return np.zeros((n_future, self.bev_h, self.bev_w), dtype=np.uint8)
        t_use = min(int(n_future), int(t_attr))
        if t_use <= 0:
            return np.zeros((n_future, self.bev_h, self.bev_w), dtype=np.uint8)

        segmentation = np.zeros((t_use, self.bev_h, self.bev_w), dtype=np.uint8)
        pedestrian = np.zeros((t_use, self.bev_h, self.bev_w), dtype=np.uint8)

        gt_agent_fut_trajs = gt_attr[:, : t_use * 2].reshape(n_agents, t_use, 2)
        gt_agent_fut_mask = gt_attr[:, t_attr * 2 : t_attr * 2 + t_use].reshape(n_agents, t_use)
        yaw_start = t_attr * 3 + 10
        gt_agent_fut_yaw = gt_attr[:, yaw_start : yaw_start + t_use].reshape(n_agents, t_use, 1)

        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        boxes = gt_boxes.copy()
        boxes[:, 6:7] = -1.0 * (boxes[:, 6:7] + np.pi / 2.0)
        gt_agent_fut_trajs = gt_agent_fut_trajs + boxes[:, None, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + boxes[:, None, 6:7]

        cls_col = t_attr * 3 + 9
        for t in range(t_use):
            for i in range(n_agents):
                if float(gt_agent_fut_mask[i, t]) < 0.5:
                    continue
                cls_idx = int(round(float(gt_attr[i, cls_col]))) if cls_col < gt_attr.shape[1] else -1
                agent_length = float(boxes[i, 4])
                agent_width = float(boxes[i, 3])
                x_a = float(gt_agent_fut_trajs[i, t, 0])
                y_a = float(gt_agent_fut_trajs[i, t, 1])
                yaw_a = float(gt_agent_fut_yaw[i, t, 0])
                poly = self._agent_poly_region(x_a, y_a, yaw_a, agent_length, agent_width)
                if cls_idx in self.vehicle_indices:
                    _fill_poly_mask(segmentation[t], poly, value=1)
                if cls_idx in self.human_indices:
                    _fill_poly_mask(pedestrian[t], poly, value=1)

        occupancy = np.logical_or(segmentation > 0, pedestrian > 0).astype(np.uint8)
        if occupancy.shape[0] < n_future:
            pad = np.zeros((n_future - occupancy.shape[0], self.bev_h, self.bev_w), dtype=np.uint8)
            occupancy = np.concatenate([occupancy, pad], axis=0)
        return occupancy[:n_future]

    def evaluate_single_coll(self, traj_xy: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        occupancy = np.asarray(occupancy, dtype=np.uint8)
        n_future = min(int(traj_xy.shape[0]), int(occupancy.shape[0]))
        if n_future <= 0:
            return np.zeros((0,), dtype=np.bool_)
        traj_xy = traj_xy[:n_future]
        occupancy = occupancy[:n_future]

        trajs = traj_xy.reshape(n_future, 1, 2).copy()
        trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
        trajs = trajs / self.dx[None, None, :]
        trajs = trajs + self._ego_rc[None, :, :]

        r = (self.bev_h - trajs[:, :, 0]).astype(np.int32)
        c = trajs[:, :, 1].astype(np.int32)
        r = np.clip(r, 0, self.bev_h - 1)
        c = np.clip(c, 0, self.bev_w - 1)

        collision = np.zeros((n_future,), dtype=np.bool_)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            valid = (rr >= 0) & (rr < self.bev_h) & (cc >= 0) & (cc < self.bev_w)
            if np.any(valid):
                collision[t] = bool(np.any(occupancy[t, rr[valid], cc[valid]] > 0))
        return collision

    def evaluate_coll(
        self,
        trajs: np.ndarray,
        gt_trajs: np.ndarray,
        occupancy: np.ndarray,
    ) -> np.ndarray:
        trajs = np.asarray(trajs, dtype=np.float32)
        gt_trajs = np.asarray(gt_trajs, dtype=np.float32)
        occupancy = np.asarray(occupancy, dtype=np.uint8)
        if trajs.ndim == 2:
            trajs = trajs[None, ...]
        if gt_trajs.ndim == 2:
            gt_trajs = gt_trajs[None, ...]
        if occupancy.ndim == 3:
            occupancy = occupancy[None, ...]

        bsz = min(int(trajs.shape[0]), int(gt_trajs.shape[0]), int(occupancy.shape[0]))
        n_future = min(int(trajs.shape[1]), int(gt_trajs.shape[1]), int(occupancy.shape[1]))
        if bsz <= 0 or n_future <= 0:
            return np.zeros((0,), dtype=np.float32)

        trajs = trajs[:bsz, :n_future]
        gt_trajs = gt_trajs[:bsz, :n_future]
        occupancy = occupancy[:bsz, :n_future]

        obj_box_coll_sum = np.zeros((n_future,), dtype=np.float32)
        ti = np.arange(n_future)

        for i in range(bsz):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], occupancy[i])
            m2 = ~gt_box_coll
            box_coll = self.evaluate_single_coll(trajs[i], occupancy[i])
            if np.any(m2):
                obj_box_coll_sum[ti[m2]] += box_coll[ti[m2]].astype(np.float32)

        return obj_box_coll_sum

    def compute_l2(self, traj_xy: np.ndarray, gt_xy: np.ndarray) -> float:
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        gt_xy = np.asarray(gt_xy, dtype=np.float32)
        n = min(int(traj_xy.shape[0]), int(gt_xy.shape[0]))
        if n <= 0:
            return float("nan")
        diff = traj_xy[:n] - gt_xy[:n]
        return float(np.linalg.norm(diff, axis=-1).mean())


def empty_planning_metric_dict() -> Dict[str, Any]:
    return {
        "planning_metrics_available": False,
        "plan_L2_1s": float("nan"),
        "plan_L2_2s": float("nan"),
        "plan_L2_3s": float("nan"),
        "plan_L2_avg": float("nan"),
        "plan_obj_box_col_1s": float("nan"),
        "plan_obj_box_col_2s": float("nan"),
        "plan_obj_box_col_3s": float("nan"),
        "plan_obj_box_col_avg": float("nan"),
    }


def compute_planning_metrics_for_scene(
    pred_xyh: np.ndarray,
    gt_xyh: np.ndarray,
    planning_labels: Optional[Dict[str, np.ndarray]],
    planning_metric: B2DPlanningMetricLite,
    target_fps: int,
) -> Dict[str, Any]:
    out = empty_planning_metric_dict()
    if planning_labels is None:
        return out

    pred_xyh = np.asarray(pred_xyh, dtype=np.float32)
    gt_xyh = np.asarray(gt_xyh, dtype=np.float32)
    n = min(int(pred_xyh.shape[0]), int(gt_xyh.shape[0]))
    if n <= 0:
        return out

    gt_boxes = np.asarray(planning_labels.get("gt_boxes"), dtype=np.float32)
    gt_attr = np.asarray(planning_labels.get("gt_attr"), dtype=np.float32)
    occupancy = planning_metric.build_occupancy(gt_boxes, gt_attr, n_future=n)
    if occupancy.shape[0] <= 0:
        return out

    n = min(n, int(occupancy.shape[0]))
    pred_xy_vehicle = pred_xyh[:n, :2]
    gt_xy_vehicle = gt_xyh[:n, :2]
    # Convert model/training vehicle frame (x forward, y left) back to
    # STP3 lidar frame (x right, y forward) for collision rasterization.
    pred_xy = np.stack([-pred_xy_vehicle[:, 1], pred_xy_vehicle[:, 0]], axis=1).astype(np.float32)
    gt_xy = np.stack([-gt_xy_vehicle[:, 1], gt_xy_vehicle[:, 0]], axis=1).astype(np.float32)
    occupancy = occupancy[:n]
    out["planning_metrics_available"] = True

    l2_values = []
    obj_box_col_values = []
    for sec in (1, 2, 3):
        horizon = min(n, max(1, int(round(float(sec) * float(target_fps)))))
        pred_h = pred_xy[:horizon]
        gt_h = gt_xy[:horizon]
        occ_h = occupancy[:horizon]
        l2_val = planning_metric.compute_l2(pred_h, gt_h)
        obj_box_col = planning_metric.evaluate_coll(
            pred_h[None, ...],
            gt_h[None, ...],
            occ_h[None, ...],
        )
        out[f"plan_L2_{sec}s"] = float(np.nan_to_num(l2_val))
        out[f"plan_obj_box_col_{sec}s"] = float(np.nan_to_num(np.mean(obj_box_col)))
        l2_values.append(out[f"plan_L2_{sec}s"])
        obj_box_col_values.append(out[f"plan_obj_box_col_{sec}s"])

    out["plan_L2_avg"] = float(np.nan_to_num(np.mean(l2_values)))
    out["plan_obj_box_col_avg"] = float(np.nan_to_num(np.mean(obj_box_col_values)))
    return out
