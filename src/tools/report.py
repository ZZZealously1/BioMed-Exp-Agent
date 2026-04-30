"""
行为报告生成模块
基于分析结果生成结构化的 Markdown 行为学报告
支持用户自定义选择报告中包含的板块
"""

from datetime import datetime
from typing import Any
import os


# 可选报告板块（key: 英文标识, value: 中文显示名）
REPORT_SECTIONS = {
    "overview": "实验概述",
    "quality": "数据质量评估",
    "metrics": "行为指标",
    "interpretation": "结果解释",
    "arena": "场地配置",
    "visualization": "可视化结果",
    "notes": "备注",
}

DEFAULT_SECTIONS = list(REPORT_SECTIONS.keys())


def generate_behavior_report(
    experiment_type: str,
    species: str | None,
    video_path: str,
    video_info: dict[str, Any],
    metrics: dict[str, Any],
    interpretation: dict[str, str],
    arena_info: dict[str, Any],
    quality_metrics: dict[str, Any] | None,
    visualization_paths: dict[str, str | None],
    user_request: str = "",
    experiment_id: str = "",
    sections: list[str] | None = None,
) -> str:
    """
    生成行为学分析报告 (Markdown 格式)

    Args:
        experiment_type: 实验类型
        species: 物种
        video_path: 视频路径
        video_info: 视频信息 (fps, width, height, duration, frame_count)
        metrics: 行为指标字典
        interpretation: 指标解释字典
        arena_info: 场地配置信息
        quality_metrics: 质量指标 (detection_rate, track_continuity)
        visualization_paths: 可视化图片路径 (trajectory_plot, heatmap)
        user_request: 用户原始请求
        experiment_id: 实验 ID
        sections: 要包含的板块列表，默认包含全部板块

    Returns:
        Markdown 格式的报告文本
    """
    if sections is None:
        sections = DEFAULT_SECTIONS.copy()
    else:
        sections = [s for s in sections if s in REPORT_SECTIONS]

    lines = []
    section_num = 0

    def add_section(title: str) -> None:
        nonlocal section_num
        section_num += 1
        lines.append(f"## {section_num}. {title}")
        lines.append("")

    # ===== 报告标题 =====
    lines.append("# 生物医学实验行为分析报告")
    lines.append("")
    if experiment_id:
        lines.append(f"**实验编号**: {experiment_id}")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ===== 实验概述 =====
    if "overview" in sections:
        add_section("实验概述")

        experiment_type_cn = {
            "open_field": "旷场实验",
            "morris_water_maze": "莫里斯水迷宫",
            "epm": "高架十字迷宫",
            "worm_assay": "线虫行为实验",
            "zebrafish_plate": "斑马鱼孔板实验",
        }.get(experiment_type, experiment_type)

        lines.append(f"| 项目 | 内容 |")
        lines.append(f"|------|------|")
        lines.append(f"| 实验类型 | {experiment_type_cn} |")
        lines.append(f"| 实验物种 | {species or '未指定'} |")
        lines.append(f"| 用户请求 | {user_request or '自动分析'} |")
        lines.append(f"| 视频文件 | {os.path.basename(video_path)} |")
        lines.append(f"| 视频时长 | {video_info.get('duration', 0):.1f} 秒 |")
        lines.append(f"| 分辨率 | {video_info.get('width', 0)} x {video_info.get('height', 0)} |")
        lines.append(f"| 帧率 | {video_info.get('fps', 0):.1f} fps |")
        lines.append(f"| 总帧数 | {video_info.get('frame_count', 0)} |")
        lines.append("")

    # ===== 数据质量评估 =====
    if "quality" in sections:
        add_section("数据质量评估")

        if quality_metrics:
            detection_rate = quality_metrics.get("detection_rate", 0)
            track_continuity = quality_metrics.get("track_continuity", 0)

            quality_level = "优秀"
            if detection_rate < 0.85 or track_continuity < 0.8:
                quality_level = "合格"
            if detection_rate < 0.7 or track_continuity < 0.6:
                quality_level = "偏低"

            lines.append(f"| 质量指标 | 数值 | 等级 |")
            lines.append(f"|----------|------|------|")
            lines.append(f"| 检测率 | {detection_rate:.1%} | {'✅ 达标' if detection_rate >= 0.85 else '⚠️ 偏低'} |")
            lines.append(f"| 跟踪连续性 | {track_continuity:.1%} | {'✅ 达标' if track_continuity >= 0.8 else '⚠️ 偏低'} |")
            lines.append(f"| **综合质量** | — | **{quality_level}** |")
            lines.append("")

            if detection_rate >= 0.85 and track_continuity >= 0.8:
                lines.append("> ✅ 数据质量良好，检测结果可靠，可用于后续行为分析。")
            elif detection_rate >= 0.7 and track_continuity >= 0.6:
                lines.append("> ⚠️ 数据质量一般，部分帧可能缺失检测，分析结果仅供参考。")
            else:
                lines.append("> ❌ 数据质量较差，检测或跟踪存在明显问题，建议重新录制视频或调整检测参数。")
            lines.append("")
        else:
            lines.append("> 未获取到质量评估数据。")
            lines.append("")

    # ===== 行为指标 =====
    if "metrics" in sections:
        add_section("行为指标")

        if experiment_type == "morris_water_maze":
            _add_water_maze_metrics(lines, metrics)
        elif experiment_type == "open_field":
            _add_open_field_metrics(lines, metrics)
        elif experiment_type == "epm":
            _add_epm_metrics(lines, metrics)
        else:
            _add_generic_metrics(lines, metrics)

    # ===== 结果解释 =====
    if "interpretation" in sections:
        add_section("结果解释")

        if interpretation:
            learning_level = interpretation.get("learning_level", "")
            learning_desc = interpretation.get("learning_description", "")
            if learning_level:
                lines.append(f"### 学习水平")
                lines.append(f"**{learning_level}** — {learning_desc}")
                lines.append("")

            path_eff = interpretation.get("path_efficiency", "")
            path_eff_desc = interpretation.get("path_efficiency_description", "")
            if path_eff:
                lines.append(f"### 路径效率")
                lines.append(f"**{path_eff}** — {path_eff_desc}")
                lines.append("")

            strategy = interpretation.get("search_strategy", "")
            strategy_desc = interpretation.get("strategy_description", "")
            if strategy:
                lines.append(f"### 搜索策略")
                lines.append(f"**{strategy}** — {strategy_desc}")
                lines.append("")

            motor_level = interpretation.get("motor_level", "")
            motor_desc = interpretation.get("motor_description", "")
            if motor_level:
                lines.append(f"### 运动能力")
                lines.append(f"**{motor_level}** — {motor_desc}")
                lines.append("")

            anxiety_level = interpretation.get("anxiety_level", "")
            anxiety_desc = interpretation.get("anxiety_description", "")
            if anxiety_level:
                lines.append(f"### 焦虑水平")
                lines.append(f"**{anxiety_level}** — {anxiety_desc}")
                lines.append("")

            summary = interpretation.get("summary", "")
            if summary:
                lines.append(f"### 综合评估")
                lines.append(f"> {summary}")
                lines.append("")
        else:
            lines.append("> 未获取到结果解释。")
            lines.append("")

    # ===== 场地配置 =====
    if "arena" in sections:
        add_section("场地配置")

        if arena_info:
            lines.append(f"| 参数 | 数值 |")
            lines.append(f"|------|------|")
            for key, value in arena_info.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        lines.append(f"| {key}.{sub_key} | {sub_value} |")
                else:
                    lines.append(f"| {key} | {value} |")
            lines.append("")
        else:
            lines.append("> 未获取到场地配置信息。")
            lines.append("")

    # ===== 可视化结果 =====
    if "visualization" in sections:
        add_section("可视化结果")

        traj_path = visualization_paths.get("trajectory_plot")
        heatmap_path = visualization_paths.get("heatmap")

        if traj_path and os.path.exists(traj_path):
            lines.append(f"### 运动轨迹图")
            lines.append(f"![轨迹图]({traj_path})")
            lines.append("")
        if heatmap_path and os.path.exists(heatmap_path):
            lines.append(f"### 热力分布图")
            lines.append(f"![热力图]({heatmap_path})")
            lines.append("")
        if not traj_path and not heatmap_path:
            lines.append("> 未生成可视化结果。")
            lines.append("")

    # ===== 备注 =====
    if "notes" in sections:
        add_section("备注")

        lines.append("- 本报告由 BioMed-Exp Agent 自动生成，基于计算机视觉分析结果。")
        lines.append("- 指标计算遵循相应实验类型的标准化协议。")
        lines.append("- 如需人工复核或进一步分析，请咨询专业研究人员。")
        lines.append("")

    return "\n".join(lines)


def _add_water_maze_metrics(lines: list[str], metrics: dict[str, Any]) -> None:
    """添加水迷宫指标"""
    metric_rows = []

    key_metrics = [
        ("escape_latency_seconds", "逃逸潜伏期", "s"),
        ("path_length", "游泳路径长度", "cm"),
        ("path_length_px", "游泳路径长度", "px"),
        ("avg_swim_speed", "平均游泳速度", "cm/s"),
        ("avg_swim_speed_px_frame", "平均游泳速度", "px/帧"),
        ("target_quadrant_time_percent", "目标象限时间占比", "%"),
        ("platform_crossings", "平台穿越次数", "次"),
        ("thigmotaxis_percent", "边缘游泳比例", "%"),
        ("avg_distance_to_platform", "平均到平台距离", "px"),
    ]

    for key, name, unit in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            metric_rows.append(f"| {name} | {value_str} | {unit} |")

    if metric_rows:
        lines.append("| 指标 | 数值 | 单位 |")
        lines.append("|------|------|------|")
        lines.extend(metric_rows)
        lines.append("")
    else:
        lines.append("> 未获取到行为指标。")
        lines.append("")


def _add_open_field_metrics(lines: list[str], metrics: dict[str, Any]) -> None:
    """添加旷场实验指标"""
    metric_rows = []

    key_metrics = [
        ("total_distance", "总移动距离", "cm"),
        ("center_time_percent", "中心区时间占比", "%"),
        ("periphery_time_percent", "边缘区时间占比", "%"),
        ("corner_time_percent", "角落区时间占比", "%"),
        ("avg_speed", "平均速度", "cm/s"),
        ("max_speed", "最大速度", "cm/s"),
        ("immobility_time_percent", "静止时间占比", "%"),
    ]

    for key, name, unit in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            metric_rows.append(f"| {name} | {value_str} | {unit} |")

    if metric_rows:
        lines.append("| 指标 | 数值 | 单位 |")
        lines.append("|------|------|------|")
        lines.extend(metric_rows)
        lines.append("")
    else:
        lines.append("> 未获取到行为指标。")
        lines.append("")


def _add_epm_metrics(lines: list[str], metrics: dict[str, Any]) -> None:
    """添加高架十字迷宫指标"""
    metric_rows = []

    key_metrics = [
        ("open_arm_time_percent", "开臂时间占比", "%"),
        ("closed_arm_time_percent", "闭臂时间占比", "%"),
        ("center_time_percent", "中央区时间占比", "%"),
        ("open_arm_entries", "开臂进入次数", "次"),
        ("closed_arm_entries", "闭臂进入次数", "次"),
        ("total_distance", "总移动距离", "cm"),
        ("avg_speed", "平均速度", "cm/s"),
    ]

    for key, name, unit in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            metric_rows.append(f"| {name} | {value_str} | {unit} |")

    if metric_rows:
        lines.append("| 指标 | 数值 | 单位 |")
        lines.append("|------|------|------|")
        lines.extend(metric_rows)
        lines.append("")
    else:
        lines.append("> 未获取到行为指标。")
        lines.append("")


def _add_generic_metrics(lines: list[str], metrics: dict[str, Any]) -> None:
    """添加通用指标"""
    if metrics:
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        for key, value in metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            lines.append(f"| {key} | {value_str} |")
        lines.append("")
    else:
        lines.append("> 未获取到行为指标。")
        lines.append("")


def generate_html_report(report_md: str, title: str = "生物医学实验行为分析报告") -> str:
    """
    将 Markdown 报告包装为带打印样式的 HTML

    Args:
        report_md: Markdown 格式的报告内容
        title: HTML 页面标题

    Returns:
        HTML 字符串
    """
    import re

    # 简单的 Markdown → HTML 转换（表格、标题、粗体、引用、列表、图片）
    html_body = report_md

    # 转义 HTML 特殊字符
    html_body = html_body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 图片 ![alt](url)
    html_body = re.sub(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        r'&lt;img src="\2" alt="\1" style="max-width:100%;height:auto;"&gt;',
        html_body
    )

    # 链接 [text](url)
    html_body = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'&lt;a href="\2"&gt;\1&lt;/a&gt;',
        html_body
    )

    # 代码块 `code`
    html_body = re.sub(
        r'`([^`]+)`',
        r'&lt;code&gt;\1&lt;/code&gt;',
        html_body
    )

    # 粗体 **text**
    html_body = re.sub(
        r'\*\*([^*]+)\*\*',
        r'&lt;strong&gt;\1&lt;/strong&gt;',
        html_body
    )

    # 标题 # ## ###
    html_body = re.sub(
        r'^# (.+)$',
        r'&lt;h1&gt;\1&lt;/h1&gt;',
        html_body,
        flags=re.MULTILINE
    )
    html_body = re.sub(
        r'^## (.+)$',
        r'&lt;h2&gt;\1&lt;/h2&gt;',
        html_body,
        flags=re.MULTILINE
    )
    html_body = re.sub(
        r'^### (.+)$',
        r'&lt;h3&gt;\1&lt;/h3&gt;',
        html_body,
        flags=re.MULTILINE
    )

    # 引用 &gt; text
    html_body = re.sub(
        r'^&gt; (.+)$',
        r'&lt;blockquote&gt;\1&lt;/blockquote&gt;',
        html_body,
        flags=re.MULTILINE
    )

    # 无序列表 - item
    html_body = re.sub(
        r'^- (.+)$',
        r'&lt;li&gt;\1&lt;/li&gt;',
        html_body,
        flags=re.MULTILINE
    )

    # 表格（简化处理：将 | 分隔的行包装为 table）
    def _wrap_table(match):
        lines = match.group(0).strip().split('\n')
        rows = []
        for i, line in enumerate(lines):
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if not cells:
                continue
            # 跳过分隔行 |---|---|
            if all(c.replace('-', '').replace(':', '') == '' for c in cells):
                continue
            tag = 'th' if i == 0 else 'td'
            row_html = ''.join(f'&lt;{tag}&gt;{c}&lt;/{tag}&gt;' for c in cells)
            rows.append(f'&lt;tr&gt;{row_html}&lt;/tr&gt;')
        return '&lt;table&gt;' + ''.join(rows) + '&lt;/table&gt;'

    # 匹配连续的表格行
    html_body = re.sub(
        r'((?:^\|[^\n]+\|\n?)+)',
        _wrap_table,
        html_body,
        flags=re.MULTILINE
    )

    # 将连续的 &lt;li&gt; 包装为 &lt;ul&gt;
    html_body = re.sub(
        r'(&lt;li&gt;.+?&lt;/li&gt;\n?)+',
        lambda m: '&lt;ul&gt;' + m.group(0) + '&lt;/ul&gt;',
        html_body
    )

    # 段落（简单的块级处理）
    paragraphs = html_body.split('\n\n')
    new_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # 如果已经是块级元素则跳过
        if p.startswith('&lt;h') or p.startswith('&lt;table') or p.startswith('&lt;ul') or p.startswith('&lt;blockquote') or p.startswith('&lt;img'):
            new_paragraphs.append(p)
        else:
            new_paragraphs.append(f'&lt;p&gt;{p}&lt;/p&gt;')
    html_body = '\n'.join(new_paragraphs)

    # 将转义的标签还原
    html_body = html_body.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* 基础样式 */
        body {{
            font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: #fff;
            padding: 40px 50px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        h1 {{
            font-size: 28px;
            color: #1a1a1a;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        h2 {{
            font-size: 20px;
            color: #1a1a1a;
            border-left: 4px solid #2563eb;
            padding-left: 12px;
            margin-top: 32px;
            margin-bottom: 16px;
        }}
        h3 {{
            font-size: 16px;
            color: #374151;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }}
        th, td {{
            border: 1px solid #d1d5db;
            padding: 10px 14px;
            text-align: left;
        }}
        th {{
            background: #f3f4f6;
            font-weight: 600;
            color: #374151;
        }}
        tr:nth-child(even) {{
            background: #f9fafb;
        }}
        blockquote {{
            border-left: 4px solid #10b981;
            background: #ecfdf5;
            padding: 12px 16px;
            margin: 12px 0;
            color: #065f46;
            border-radius: 0 6px 6px 0;
        }}
        ul {{
            padding-left: 24px;
            margin: 12px 0;
        }}
        li {{
            margin: 6px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            margin: 16px 0;
        }}
        code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 13px;
        }}
        strong {{
            color: #1a1a1a;
        }}
        p {{
            margin: 10px 0;
        }}
        .print-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 24px;
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}
        .print-btn:hover {{
            background: #1d4ed8;
        }}

        /* 打印样式 */
        @media print {{
            body {{
                background: #fff;
                margin: 0;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 20px 30px;
                border-radius: 0;
            }}
            .print-btn {{
                display: none;
            }}
            h1 {{
                font-size: 22px;
            }}
            h2 {{
                font-size: 16px;
                page-break-after: avoid;
            }}
            h3 {{
                font-size: 14px;
                page-break-after: avoid;
            }}
            table {{
                page-break-inside: avoid;
            }}
            tr {{
                page-break-inside: avoid;
            }}
            img {{
                page-break-inside: avoid;
            }}
            blockquote {{
                page-break-inside: avoid;
            }}
            /* 页眉页脚 */
            @page {{
                margin: 20mm 15mm;
                @top-center {{
                    content: "{title}";
                    font-size: 10px;
                    color: #999;
                }}
                @bottom-center {{
                    content: "第 " counter(page) " 页";
                    font-size: 10px;
                    color: #999;
                }}
            }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ 打印 / 另存为 PDF</button>
    <div class="container">
{html_body}
    </div>
</body>
</html>"""
    return html_template
