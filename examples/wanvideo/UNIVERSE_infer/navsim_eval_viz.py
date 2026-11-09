from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

HISTORY_TRAJ_COLOR = "#1f77b4"
HISTORY_LINK_COLOR = "#6b7280"


def _to_pil_frames(video: List[Any], width: int, height: int) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for frame in video:
        if isinstance(frame, Image.Image):
            img = frame.convert("RGB")
        elif isinstance(frame, np.ndarray):
            img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        elif torch.is_tensor(frame):
            arr = frame.detach().cpu().float()
            if arr.ndim == 3:
                arr = arr.permute(1, 2, 0)
            arr = (arr + 1.0) * 127.5 if arr.min() < 0 else arr * 255.0
            arr = arr.clamp(0, 255).to(torch.uint8).numpy()
            img = Image.fromarray(arr).convert("RGB")
        else:
            continue
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        frames.append(img)
    return frames


def _render_traj_plot(
    history_traj: np.ndarray,
    gt_future_traj: np.ndarray,
    pred_future_traj: np.ndarray,
    width: int,
    height: int,
    score_dict: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for --save_viz trajectory plotting.") from exc

    hist = np.asarray(history_traj, dtype=np.float32)
    gt = np.asarray(gt_future_traj, dtype=np.float32)
    pred = np.asarray(pred_future_traj, dtype=np.float32)
    hist_xy = hist[:, :2] if hist.ndim == 2 and hist.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    gt_xy = gt[:, :2] if gt.ndim == 2 and gt.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    pred_xy = pred[:, :2] if pred.ndim == 2 and pred.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)

    dpi = 100
    fig_w = max(1.0, width / dpi)
    fig_h = max(1.0, height / dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if hist_xy.shape[0] > 0:
        ax.plot(hist_xy[:, 0], hist_xy[:, 1], "-o", color=HISTORY_TRAJ_COLOR, linewidth=2, markersize=4, label="History GT")
    if gt_xy.shape[0] > 0:
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-o", color="#2ca02c", linewidth=2, markersize=4, label="GT Future")
    if pred_xy.shape[0] > 0:
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "--o", color="#d62728", linewidth=2, markersize=4, label="Pred Future")

    # Current ego pose anchor at t=0 for continuity in the timeline.
    ax.plot([0.0], [0.0], marker="*", markersize=10, color="#222222", label="Current")
    if hist_xy.shape[0] > 0:
        ax.plot([hist_xy[-1, 0], 0.0], [hist_xy[-1, 1], 0.0], ":", color=HISTORY_LINK_COLOR, linewidth=1.5)
    if gt_xy.shape[0] > 0:
        ax.plot([0.0, gt_xy[0, 0]], [0.0, gt_xy[0, 1]], ":", color="#2ca02c", linewidth=1.5, alpha=0.8)
    if pred_xy.shape[0] > 0:
        ax.plot([0.0, pred_xy[0, 0]], [0.0, pred_xy[0, 1]], ":", color="#d62728", linewidth=1.5, alpha=0.8)

    all_pts = [arr for arr in (hist_xy, gt_xy, pred_xy, np.array([[0.0, 0.0]], dtype=np.float32)) if arr.shape[0] > 0]
    if all_pts:
        pts = np.concatenate(all_pts, axis=0)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        span = max(float(x_max - x_min), float(y_max - y_min), 1e-3)
        pad = 0.15 * span
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.canvas.draw()

    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = np.ascontiguousarray(rgba[:, :, :3])
    else:
        plot_w, plot_h = fig.canvas.get_width_height()
        if hasattr(fig.canvas, "tostring_rgb"):
            rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(plot_h, plot_w, 3)
        elif hasattr(fig.canvas, "tostring_argb"):
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(plot_h, plot_w, 4)
            rgb = np.ascontiguousarray(argb[:, :, 1:4])
        else:
            plt.close(fig)
            raise RuntimeError("Unsupported matplotlib canvas backend for trajectory plotting.")
    plt.close(fig)
    img = Image.fromarray(rgb).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return img


def format_viz_index_prefix(index: Optional[int], width: int = 6) -> str:
    if index is None:
        return ""
    try:
        idx = int(index)
    except Exception:
        return ""
    if idx <= 0:
        return ""
    return f"{idx:0{max(1, int(width))}d}_"


def _camera_attr(camera_obj: Any, key: str) -> Any:
    if camera_obj is None:
        return None
    if isinstance(camera_obj, dict):
        return camera_obj.get(key)
    return getattr(camera_obj, key, None)


def _camera_hw(camera_obj: Any, fallback_h: int, fallback_w: int) -> tuple[int, int]:
    image_value = _camera_attr(camera_obj, "image")
    if isinstance(image_value, np.ndarray) and image_value.ndim >= 2:
        h, w = int(image_value.shape[0]), int(image_value.shape[1])
        if h > 1 and w > 1:
            return h, w
    if isinstance(image_value, (str, Path)):
        try:
            with Image.open(image_value) as im:
                w, h = im.size
            if h > 1 and w > 1:
                return int(h), int(w)
        except Exception:
            pass
    return int(fallback_h), int(fallback_w)


def _scale_intrinsics(intrinsic: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int) -> np.ndarray:
    k = np.asarray(intrinsic, dtype=np.float64).copy()
    if k.ndim != 2 or k.shape[0] < 3 or k.shape[1] < 3:
        return k
    src_h = max(1, int(src_h))
    src_w = max(1, int(src_w))
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def _fit_traj_xy(traj: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32, copy=False)


def _project_local_xy_to_image(
    local_xy: np.ndarray,
    camera_obj: Any,
    image_h: int,
    image_w: int,
    eps: float = 1e-3,
    allow_behind_camera: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    local_xy = _fit_traj_xy(local_xy)
    if local_xy.shape[0] <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)

    s2l_r_raw = _camera_attr(camera_obj, "sensor2lidar_rotation")
    s2l_t_raw = _camera_attr(camera_obj, "sensor2lidar_translation")
    intrinsic_raw = _camera_attr(camera_obj, "intrinsics")
    if s2l_r_raw is None or s2l_t_raw is None or intrinsic_raw is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)

    try:
        s2l_r = np.asarray(s2l_r_raw, dtype=np.float64)
        s2l_t = np.asarray(s2l_t_raw, dtype=np.float64).reshape(-1)[:3]
        intrinsic = np.asarray(intrinsic_raw, dtype=np.float64)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)

    if s2l_r.shape != (3, 3) or s2l_t.shape[0] < 3 or intrinsic.ndim != 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)

    src_h, src_w = _camera_hw(camera_obj, fallback_h=image_h, fallback_w=image_w)
    intrinsic_scaled = _scale_intrinsics(intrinsic, src_h=src_h, src_w=src_w, dst_h=image_h, dst_w=image_w)

    points_lidar = np.concatenate(
        [
            local_xy.astype(np.float64, copy=False),
            np.zeros((local_xy.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )

    # sensor2lidar gives camera->lidar transform.
    # For row vectors: p_cam = (p_lidar - t_sensor2lidar) @ R_sensor2lidar
    points_cam = (points_lidar - s2l_t[None, :]) @ s2l_r
    points_img_h = points_cam @ intrinsic_scaled.T
    proj_depth = points_img_h[:, 2]
    if allow_behind_camera:
        abs_depth = np.abs(proj_depth)
        safe_depth = np.where(
            abs_depth > float(eps),
            proj_depth,
            np.where(proj_depth >= 0.0, float(eps), -float(eps)),
        )
        valid_depth = abs_depth > float(eps)
    else:
        safe_depth = np.maximum(proj_depth, float(eps))
        valid_depth = proj_depth > float(eps)
    points_img = points_img_h[:, :2] / safe_depth[:, None]

    in_image = (
        (points_img[:, 0] >= 0.0)
        & (points_img[:, 0] <= max(0, image_w - 1))
        & (points_img[:, 1] >= 0.0)
        & (points_img[:, 1] <= max(0, image_h - 1))
    )
    finite = np.isfinite(points_img).all(axis=1)
    valid = valid_depth & in_image & finite
    return points_img.astype(np.float32, copy=False), valid.astype(np.bool_, copy=False)


def _draw_projected_polyline(
    img: Image.Image,
    points: np.ndarray,
    valid_mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int,
    draw_points: bool = True,
) -> None:
    if points.ndim != 2 or points.shape[0] <= 1:
        return
    draw = ImageDraw.Draw(img)
    for i in range(1, points.shape[0]):
        if bool(valid_mask[i - 1]) and bool(valid_mask[i]):
            p0 = (float(points[i - 1, 0]), float(points[i - 1, 1]))
            p1 = (float(points[i, 0]), float(points[i, 1]))
            draw.line([p0, p1], fill=color, width=max(1, int(width)))
    if draw_points:
        radius = max(1, int(round(width * 0.8)))
        for i in range(points.shape[0]):
            if not bool(valid_mask[i]):
                continue
            x, y = float(points[i, 0]), float(points[i, 1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _draw_projected_marker(
    img: Image.Image,
    point: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int = 5,
    outline_color: tuple[int, int, int] = (255, 255, 255),
    outline_width: int = 2,
) -> None:
    if point.ndim != 1 or point.shape[0] < 2:
        return
    x, y = float(point[0]), float(point[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return
    draw = ImageDraw.Draw(img)
    r = max(1, int(radius))
    ow = max(0, int(outline_width))
    if ow > 0:
        ro = r + ow
        draw.ellipse((x - ro, y - ro, x + ro, y + ro), fill=outline_color)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _render_overlay_current_frame(
    frame: Image.Image,
    *,
    camera_for_projection: Any,
    gt_future_traj: np.ndarray,
    pred_future_traj: np.ndarray,
    history_traj: np.ndarray,
) -> Image.Image:
    out = frame.convert("RGB").copy()
    h, w = out.height, out.width

    gt_xy = _fit_traj_xy(gt_future_traj)
    pred_xy = _fit_traj_xy(pred_future_traj)
    hist_xy = _fit_traj_xy(history_traj)
    anchor = np.array([[0.0, 0.0]], dtype=np.float32)

    gt_line = np.concatenate([anchor, gt_xy], axis=0) if gt_xy.shape[0] > 0 else anchor.copy()
    pred_line = np.concatenate([anchor, pred_xy], axis=0) if pred_xy.shape[0] > 0 else anchor.copy()
    hist_line = np.concatenate([hist_xy, anchor], axis=0) if hist_xy.shape[0] > 0 else anchor.copy()

    hist_proj, hist_valid = _project_local_xy_to_image(hist_line, camera_for_projection, h, w)
    gt_proj, gt_valid = _project_local_xy_to_image(gt_line, camera_for_projection, h, w)
    pred_proj, pred_valid = _project_local_xy_to_image(pred_line, camera_for_projection, h, w)

    _draw_projected_polyline(out, hist_proj, hist_valid, color=(31, 119, 180), width=3, draw_points=False)
    _draw_projected_polyline(out, gt_proj, gt_valid, color=(44, 160, 44), width=4, draw_points=True)
    _draw_projected_polyline(out, pred_proj, pred_valid, color=(214, 39, 40), width=4, draw_points=True)
    return out


def save_bev_with_agent_artifacts(
    *,
    out_dir: Path,
    token: str,
    scene: Any,
    pred_future_traj: np.ndarray,
    rank: int,
    suffix: str = "",
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    import matplotlib.pyplot as plt

    pred_xyh = np.asarray(pred_future_traj, dtype=np.float32)
    if pred_xyh.ndim != 2 or pred_xyh.shape[0] <= 0:
        raise ValueError("pred_future_traj must be [T, C] with T > 0.")
    if pred_xyh.shape[1] < 3:
        pad = np.zeros((pred_xyh.shape[0], 3 - pred_xyh.shape[1]), dtype=np.float32)
        pred_xyh = np.concatenate([pred_xyh, pad], axis=1)
    pred_xyh = pred_xyh[:, :3].astype(np.float32, copy=False)

    sample_interval_s = 0.5
    try:
        sample_interval_s = float(getattr(scene.get_future_trajectory(), "trajectory_sampling").interval_length)
    except Exception:
        sample_interval_s = 0.5

    pred_traj = Trajectory(
        poses=pred_xyh,
        trajectory_sampling=TrajectorySampling(num_poses=int(pred_xyh.shape[0]), interval_length=float(sample_interval_s)),
    )

    class _FixedTrajectoryAgent:
        def __init__(self, traj_obj: Any) -> None:
            self._traj_obj = traj_obj

        def compute_trajectory(self, _agent_input: Any) -> Any:
            return self._traj_obj

        def compute_trajectory_vis(self, _agent_input: Any) -> Any:
            return self._traj_obj

    tag = f"{token}{suffix}_rank{rank}"
    png_path = out_dir / f"{tag}.png"
    gif_path = out_dir / f"{tag}.gif"

    def _plot_bev_with_agent_compatible(scene_obj: Any, agent_obj: Any) -> Any:
        from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
        from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG

        human_trajectory = scene_obj.get_future_trajectory()
        agent_trajectory = agent_obj.compute_trajectory(scene_obj.get_agent_input())

        frame_idx = int(scene_obj.scene_metadata.num_history_frames) - 1
        frame_idx = max(0, min(frame_idx, len(scene_obj.frames) - 1))
        fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
        add_configured_bev_on_ax(ax, scene_obj.map_api, scene_obj.frames[frame_idx])
        add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
        add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])

        margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
        ax.set_aspect("equal")
        ax.set_xlim(-margin_y / 2, margin_y / 2)
        ax.set_ylim(-margin_x / 2, margin_x / 2)
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    agent = _FixedTrajectoryAgent(pred_traj)
    fig, _ax = _plot_bev_with_agent_compatible(scene, agent)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    with Image.open(png_path) as img_raw:
        img = img_raw.convert("RGB")
        img.save(gif_path, format="GIF", save_all=True, append_images=[img.copy()], duration=700, loop=0)
    return {"image_path": str(png_path), "gif_path": str(gif_path)}


def save_viz_video(
    out_path: Path,
    gt_video: List[Any],
    pred_video: List[Any],
    history_traj: np.ndarray,
    gt_future_traj: np.ndarray,
    pred_future_traj: np.ndarray,
    width: int,
    height: int,
    plot_height: int,
    fps: int,
    score_dict: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    camera_for_projection: Optional[Any] = None,
    num_history_frames: Optional[int] = None,
    gt_future_max_steps: Optional[int] = None,
    projection_image_path: Optional[Path] = None,
    projection_overlay_on_video: bool = False,
) -> None:
    gt_frames = _to_pil_frames(gt_video, width=width, height=height)
    pred_frames = _to_pil_frames(pred_video, width=width, height=height)
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        raise ValueError("Cannot build visualization: empty GT or prediction video.")

    gt_future_viz_traj = np.asarray(gt_future_traj, dtype=np.float32)
    if gt_future_viz_traj.ndim != 2:
        gt_future_viz_traj = np.zeros((0, 3), dtype=np.float32)
    if gt_future_max_steps is not None:
        max_steps = max(0, int(gt_future_max_steps))
        gt_future_viz_traj = gt_future_viz_traj[:max_steps]

    inferred_hist = None
    hist_arr = np.asarray(history_traj, dtype=np.float32)
    if hist_arr.ndim == 2 and hist_arr.shape[0] > 0:
        inferred_hist = int(hist_arr.shape[0] + 1)
    history_count = int(num_history_frames) if num_history_frames is not None else int(inferred_hist or 1)
    current_idx = max(0, min(history_count - 1, len(gt_frames) - 1, len(pred_frames) - 1))

    if projection_image_path is not None and camera_for_projection is not None:
        gt_current = gt_frames[current_idx]
        pred_current = pred_frames[current_idx]
        gt_overlay = _render_overlay_current_frame(
            gt_current,
            camera_for_projection=camera_for_projection,
            gt_future_traj=gt_future_viz_traj,
            pred_future_traj=pred_future_traj,
            history_traj=history_traj,
        )
        pred_overlay = _render_overlay_current_frame(
            pred_current,
            camera_for_projection=camera_for_projection,
            gt_future_traj=gt_future_viz_traj,
            pred_future_traj=pred_future_traj,
            history_traj=history_traj,
        )
        overlay_canvas = Image.new("RGB", (width * 2, height), (255, 255, 255))
        overlay_canvas.paste(gt_overlay, (0, 0))
        overlay_canvas.paste(pred_overlay, (width, 0))
        draw = ImageDraw.Draw(overlay_canvas)
        draw.rectangle((0, 0, 270, 26), fill=(255, 255, 255))
        draw.rectangle((width, 0, width + 290, 26), fill=(255, 255, 255))
        draw.text((8, 5), "GT current + projected traj", fill=(0, 0, 0))
        draw.text((width + 8, 5), "Pred current + projected traj", fill=(0, 0, 0))
        projection_image_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_canvas.save(projection_image_path)

        if projection_overlay_on_video:
            canvas_w = width
            canvas_h = height * 2 + plot_height
            traj_plot = _render_traj_plot(
                history_traj=history_traj,
                gt_future_traj=gt_future_viz_traj,
                pred_future_traj=pred_future_traj,
                width=canvas_w,
                height=plot_height,
                score_dict=score_dict,
                extra_info=extra_info,
            )
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(max(1, fps)),
                (canvas_w, canvas_h),
            )
            n = max(1, len(pred_frames))
            for _ in range(n):
                canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
                canvas.paste(gt_overlay, (0, 0))
                canvas.paste(pred_overlay, (0, height))
                canvas.paste(traj_plot, (0, height * 2))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 230, 26), fill=(255, 255, 255))
                draw.rectangle((0, height, 230, height + 26), fill=(255, 255, 255))
                draw.text((8, 5), "GT current + projected traj", fill=(0, 0, 0))
                draw.text((8, height + 5), "Pred current + projected traj", fill=(0, 0, 0))
                writer.write(np.array(canvas)[:, :, ::-1])
            writer.release()
            return

    # Default visualization layout: GT/Pred videos side-by-side + trajectory plot.
    canvas_w = width * 2
    canvas_h = height + plot_height
    traj_plot = _render_traj_plot(
        history_traj=history_traj,
        gt_future_traj=gt_future_viz_traj,
        pred_future_traj=pred_future_traj,
        width=canvas_w,
        height=plot_height,
        score_dict=score_dict,
        extra_info=extra_info,
    )

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, fps)),
        (canvas_w, canvas_h),
    )
    n = max(len(gt_frames), len(pred_frames))
    for i in range(n):
        gt_frame = gt_frames[min(i, len(gt_frames) - 1)]
        pred_frame = pred_frames[min(i, len(pred_frames) - 1)]
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(gt_frame, (0, 0))
        canvas.paste(pred_frame, (width, 0))
        canvas.paste(traj_plot, (0, height))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 170, 26), fill=(255, 255, 255))
        draw.rectangle((width, 0, width + 210, 26), fill=(255, 255, 255))
        draw.text((8, 5), "GT Video", fill=(0, 0, 0))
        draw.text((width + 8, 5), "Pred Video", fill=(0, 0, 0))
        writer.write(np.array(canvas)[:, :, ::-1])
    writer.release()
