"""
可视化工具
生成轨迹图和热力图
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

from scipy.stats import gaussian_kde


def generate_trajectory_plot(
    positions: np.ndarray,
    arena_info: dict,
    experiment_type: str,
    video_size: tuple[int, int],
    output_path: str,
    title: str | None = None,
    skeletons: list[dict] | None = None,
    tracks: list[dict] | None = None,
) -> str:
    """
    生成轨迹图

    Args:
        positions: (N, 2) 像素坐标数组
        arena_info: 场地信息字典
        experiment_type: 实验类型
        video_size: (width, height)
        output_path: 输出文件路径
        title: 图表标题
        skeletons: 线虫骨架数据列表 (用于 worm 模式叠加骨架线)
        tracks: 轨迹列表，用于多目标独立绘制 [{track_id, positions: [{x, y, frame_idx}]}]

    Returns:
        输出文件路径
    """
    width, height = video_size

    # 统一图尺寸（与热力图保持一致）
    aspect = height / width if width > 0 else 1.0
    fig_w = 10
    fig_h = max(6, fig_w * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # 绘制场地轮廓
    _draw_arena_overlay(ax, arena_info, experiment_type)

    # 多目标模式 (worm_assay / zebrafish_plate)
    if tracks and experiment_type in ("worm_assay", "zebrafish_plate") and len(tracks) > 1:
        # 按原始 track_id 排序，并重新映射为连续的 1-based ID
        sorted_tracks = sorted(tracks, key=lambda t: t.get("track_id", 0))
        id_map = {t.get("track_id", i): i + 1 for i, t in enumerate(sorted_tracks)}
        cmap = plt.cm.get_cmap('tab10', max(len(sorted_tracks), 10))
        for idx, traj in enumerate(sorted_tracks):
            traj_positions = np.array(
                [[p["x"], p["y"]] for p in traj.get("positions", [])],
                dtype=np.float32
            )
            if len(traj_positions) < 2:
                continue
            color = cmap(idx % 10)
            tid = id_map.get(traj.get("track_id", idx + 1), idx + 1)
            # 轨迹线
            ax.plot(
                traj_positions[:, 0], traj_positions[:, 1],
                '-', color=color, linewidth=1.5, alpha=0.8,
                label=f'Track {tid}', zorder=3
            )
            # 起点
            ax.plot(
                traj_positions[0, 0], traj_positions[0, 1],
                'o', color=color, markersize=5,
                markeredgecolor='black', markeredgewidth=0.5,
                zorder=5
            )
            # 终点
            ax.plot(
                traj_positions[-1, 0], traj_positions[-1, 1],
                's', color=color, markersize=5,
                markeredgecolor='black', markeredgewidth=0.5,
                zorder=5
            )
        # 统一起点/终点图例说明
        ax.plot([], [], 'o', color='gray', markersize=5,
                markeredgecolor='black', markeredgewidth=0.5,
                label='Start', zorder=0)
        ax.plot([], [], 's', color='gray', markersize=5,
                markeredgecolor='black', markeredgewidth=0.5,
                label='End', zorder=0)
    else:
        # 降采样 (如果点太多)
        if len(positions) > 2000:
            indices = np.linspace(0, len(positions) - 1, 2000, dtype=int)
            plot_positions = positions[indices]
        else:
            plot_positions = positions

        # 绘制轨迹 (按时间渐变颜色)
        if len(plot_positions) > 1:
            points = plot_positions.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, len(segments))
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=1.0, alpha=0.8)
            lc.set_array(np.arange(len(segments)))
            line = ax.add_collection(lc)
            # 颜色条：时间进度
            cbar = fig.colorbar(line, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label("Time progress", fontsize=9)

        # 起点 (绿色) 和终点 (红色)
        if len(positions) > 0:
            ax.plot(positions[0, 0], positions[0, 1], 'o', color='limegreen',
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                    label='Start', zorder=5)
            ax.plot(positions[-1, 0], positions[-1, 1], 'o', color='red',
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                    label='End', zorder=5)

    # 标题和图例
    if title:
        ax.set_title(title, fontsize=14)
    else:
        type_names = {
            "open_field": "Open Field Test",
            "epm": "Elevated Plus Maze",
            "morris_water_maze": "Morris Water Maze",
            "worm_assay": "C. elegans Behavior Assay",
            "zebrafish_plate": "Zebrafish Plate Assay"
        }
        ax.set_title(f"Trajectory - {type_names.get(experiment_type, experiment_type)}", fontsize=14)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8,
              title="Tracks" if (tracks and experiment_type in ("worm_assay", "zebrafish_plate") and len(tracks) > 1) else None)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")

    # 右侧添加固定宽度的占位区域，确保与热力图 colorbar 区域等宽
    divider = make_axes_locatable(ax)
    dummy_ax = divider.append_axes("right", size="4%", pad=0.15)
    dummy_ax.set_visible(False)

    fig.savefig(output_path, dpi=150, bbox_inches=None, facecolor='white')
    plt.close(fig)

    return output_path


def generate_heatmap(
    positions: np.ndarray,
    arena_info: dict,
    experiment_type: str,
    video_size: tuple[int, int],
    output_path: str,
    title: str | None = None,
    colormap: str = "jet",
) -> str:
    """
    生成空间密度热力图

    Args:
        positions: (N, 2) 像素坐标数组
        arena_info: 场地信息字典
        experiment_type: 实验类型
        video_size: (width, height)
        output_path: 输出文件路径
        title: 图表标题
        colormap: 颜色映射

    Returns:
        输出文件路径
    """
    width, height = video_size

    # 统一图尺寸（与轨迹图保持一致）
    aspect = height / width if width > 0 else 1.0
    fig_w = 10
    fig_h = max(6, fig_w * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    if len(positions) >= 10:
        # 使用 Gaussian KDE 计算密度
        kde = gaussian_kde(positions.T, bw_method='scott')

        # 在网格上评估
        grid_n = 200
        x_grid = np.linspace(0, width, grid_n)
        y_grid = np.linspace(0, height, grid_n)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(grid_points).reshape(grid_n, grid_n)

        # 显示热力图
        im = ax.imshow(Z, extent=[0, width, height, 0],
                       cmap=colormap, alpha=0.7, aspect='auto')
        # 右侧固定宽度的 colorbar，与轨迹图 legend 区域等宽
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        fig.colorbar(im, cax=cax, label='Density')
    else:
        # 点太少，用散点图代替
        if len(positions) > 0:
            ax.scatter(positions[:, 0], positions[:, 1], s=5, alpha=0.5, c='blue')

    # 绘制场地轮廓（斑马鱼孔板实验不绘制，避免干扰热力图）
    if experiment_type != "zebrafish_plate":
        _draw_arena_overlay(ax, arena_info, experiment_type, fill=False)

    # 标题
    if title:
        ax.set_title(title, fontsize=14)
    else:
        type_names = {
            "open_field": "Open Field Test",
            "epm": "Elevated Plus Maze",
            "morris_water_maze": "Morris Water Maze",
            "zebrafish_plate": "Zebrafish Plate Assay"
        }
        ax.set_title(f"Heatmap - {type_names.get(experiment_type, experiment_type)}", fontsize=14)

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")

    fig.savefig(output_path, dpi=150, bbox_inches=None, facecolor='white')
    plt.close(fig)

    return output_path


def _draw_arena_overlay(
    ax,
    arena_info: dict,
    experiment_type: str,
    fill: bool = True,
    alpha: float = 0.3,
) -> None:
    """绘制场地轮廓叠加"""
    if not arena_info:
        return

    if experiment_type == "open_field":
        _draw_open_field_arena(ax, arena_info, fill, alpha)
    elif experiment_type == "zebrafish_plate":
        _draw_zebrafish_arena(ax, arena_info, alpha)
    elif experiment_type == "epm":
        _draw_epm_arena(ax, arena_info, fill, alpha)
    elif experiment_type == "morris_water_maze":
        _draw_water_maze_arena(ax, arena_info, fill, alpha)


def _draw_open_field_arena(ax, arena_info: dict, fill: bool, alpha: float):
    """绘制旷场实验场地"""
    cx = arena_info.get("center_x", 0)
    cy = arena_info.get("center_y", 0)
    center_radius = arena_info.get("center_radius", 0)
    edge_width = arena_info.get("edge_width", 0)

    w = arena_info.get("width", 0)
    h = arena_info.get("height", 0)
    arena_radius = min(w, h) / 2

    # 整个场地圆
    arena_circle = plt.Circle((cx, cy), arena_radius, fill=False,
                               edgecolor='gray', linewidth=1.5, linestyle='-')
    ax.add_patch(arena_circle)

    # 中心区
    center_circle = plt.Circle((cx, cy), center_radius, fill=fill,
                                facecolor='green', edgecolor='green',
                                linestyle='--', linewidth=1.0, alpha=alpha)
    ax.add_patch(center_circle)

    # 边缘带 (外圆 - 内圆)
    if edge_width > 0:
        edge_inner = plt.Circle((cx, cy), arena_radius - edge_width, fill=False,
                                 edgecolor='orange', linestyle=':', linewidth=1.0, alpha=alpha)
        ax.add_patch(edge_inner)


def _draw_zebrafish_arena(ax, arena_info: dict, alpha: float):
    """绘制斑马鱼孔板实验的孔位轮廓"""
    wells = arena_info.get("wells", [])
    if not wells:
        return

    for well in wells:
        cx = well.get("center_x", well.get("x", 0))
        cy = well.get("center_y", well.get("y", 0))
        radius = well.get("radius", well.get("r", 0))
        if radius <= 0:
            continue
        # 绘制孔边界（空心圆）
        circle = plt.Circle(
            (cx, cy), radius,
            fill=False,
            edgecolor='gray', linewidth=1.0, linestyle='-',
            alpha=0.5
        )
        ax.add_patch(circle)


def _draw_epm_arena(ax, arena_info: dict, fill: bool, alpha: float):
    """绘制高架十字迷宫场地"""
    cx = arena_info.get("center_x", 0)
    cy = arena_info.get("center_y", 0)
    arm_width = arena_info.get("arm_width", 0)
    arm_length = arena_info.get("arm_length", 0)
    center_size = arena_info.get("center_size", arm_width)

    half_aw = arm_width / 2
    half_cs = center_size / 2

    # 中央区
    ax.add_patch(patches.Rectangle(
        (cx - half_cs, cy - half_cs), center_size, center_size,
        fill=fill, facecolor='lightgray', edgecolor='gray',
        linewidth=1.5, alpha=alpha
    ))

    # 开臂 (水平方向 — 左右)
    # 右臂
    ax.add_patch(patches.Rectangle(
        (cx + half_cs, cy - half_aw), arm_length, arm_width,
        fill=fill, facecolor='lightgreen', edgecolor='green',
        linewidth=1.0, linestyle='--', alpha=alpha, label='Open Arm'
    ))
    # 左臂
    ax.add_patch(patches.Rectangle(
        (cx - half_cs - arm_length, cy - half_aw), arm_length, arm_width,
        fill=fill, facecolor='lightgreen', edgecolor='green',
        linewidth=1.0, linestyle='--', alpha=alpha
    ))

    # 闭臂 (垂直方向 — 上下)
    # 上臂 (y 减小方向)
    ax.add_patch(patches.Rectangle(
        (cx - half_aw, cy - half_cs - arm_length), arm_width, arm_length,
        fill=fill, facecolor='lightsalmon', edgecolor='red',
        linewidth=1.0, alpha=alpha, label='Closed Arm'
    ))
    # 下臂 (y 增大方向)
    ax.add_patch(patches.Rectangle(
        (cx - half_aw, cy + half_cs), arm_width, arm_length,
        fill=fill, facecolor='lightsalmon', edgecolor='red',
        linewidth=1.0, alpha=alpha
    ))


def _draw_water_maze_arena(ax, arena_info: dict, fill: bool, alpha: float):
    """绘制水迷宫场地"""
    pool_d = arena_info.get("pool_diameter", 0)
    pool_center = arena_info.get("pool_center", {})
    platform_center = arena_info.get("platform_center", {})
    platform_radius = arena_info.get("platform_radius", 0)

    pcx = pool_center.get("x", 0)
    pcy = pool_center.get("y", 0)
    pool_radius = pool_d / 2

    # 水池
    pool = plt.Circle((pcx, pcy), pool_radius, fill=fill,
                        facecolor='lightblue', edgecolor='blue',
                        linewidth=2.0, alpha=alpha * 0.5)
    ax.add_patch(pool)

    # 象限分割线
    ax.plot([pcx - pool_radius, pcx + pool_radius], [pcy, pcy],
            'k--', alpha=0.3, linewidth=0.8)
    ax.plot([pcx, pcx], [pcy - pool_radius, pcy + pool_radius],
            'k--', alpha=0.3, linewidth=0.8)

    # 平台
    if platform_radius > 0:
        plat_x = platform_center.get("x", 0)
        plat_y = platform_center.get("y", 0)
        platform = plt.Circle((plat_x, plat_y), platform_radius, fill=fill,
                               facecolor='red', edgecolor='red',
                               linewidth=1.5, alpha=0.5, label='Platform')
        ax.add_patch(platform)
