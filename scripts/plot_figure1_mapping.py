import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def draw_figure1_professional():
    # --- Professional Settings ---
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig = plt.figure(figsize=(16, 8), dpi=300)
    
    # Professional Color Palette (Muted/Technical)
    colors = {
        'col_corner': '#2C3E50',     # Navy Gray
        'col_edge': '#34495E',       # Steel Blue
        'col_interior': '#7F8C8D',   # Gray
        'beam_exterior': '#C0392B',  # Deep Red
        'beam_interior': '#E67E22',  # Burnt Orange
        'gene_bg': '#ECF0F1',        # Background Light Gray
        'rotation': '#95A5A6',       # Silver
        'grid': '#BDC3C7'            # Light Grid
    }

    # --- 1. 3D Frame View (Left) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("(a) 3D Frame Member Grouping (Hybrid Strategy)", fontsize=16, pad=30, fontweight='bold')

    # Draw a subtle ground grid
    grid_size = np.linspace(0, 10, 3)
    for x in grid_size:
        ax1.plot([x, x], [0, 10], [0, 0], color=colors['grid'], lw=0.8, alpha=0.5, zorder=0)
    for y in grid_size:
        ax1.plot([0, 10], [y, y], [0, 0], color=colors['grid'], lw=0.8, alpha=0.5, zorder=0)

    # Node generation
    nodes = {}
    idx = 0
    for z in [0, 4]:
        for y in [0, 5, 10]:
            for x in [0, 5, 10]:
                nodes[idx] = (x, y, z)
                idx += 1

    # Draw Columns
    for i in range(9):
        p1, p2 = nodes[i], nodes[i+9]
        x, y = p1[0], p1[1]
        if (x in [0, 10]) and (y in [0, 10]): color = colors['col_corner']
        elif (x in [0, 10]) or (y in [0, 10]): color = colors['col_edge']
        else: color = colors['col_interior']
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=7, solid_capstyle='round', zorder=5)

    # Draw Beams
    x_beams = [(9,10), (10,11), (12,13), (13,14), (15,16), (16,17)]
    y_beams = [(9,12), (12,15), (10,13), (13,16), (11,14), (14,17)]
    
    for start, end in x_beams + y_beams:
        p1, p2 = nodes[start], nodes[end]
        is_ext = (p1[0]==p2[0] and p1[0] in [0, 10]) or (p1[1]==p2[1] and p1[1] in [0, 10])
        color = colors['beam_exterior'] if is_ext else colors['beam_interior']
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=5, alpha=0.8, solid_capstyle='round', zorder=10)

    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=40)
    ax1.set_box_aspect((1, 1, 0.8)) # Adjust aspect ratio

    # --- 2. Chromosome Structure (Right) ---
    ax2 = fig.add_subplot(122)
    ax2.set_title("(b) Genetic Chromosome Mapping Mechanism", fontsize=16, pad=30, fontweight='bold')

    # Mapping blocks setup
    gene_struct = [
        (r'$C_{id,1}$', colors['col_corner'], 'Corner'),
        (r'$C_{id,2}$', colors['col_edge'], 'Edge'),
        (r'$C_{id,3}$', colors['col_interior'], 'Interior'),
        (r'$R_{dir,1}$', colors['rotation'], '0 or 1'),
        (r'$R_{dir,2}$', colors['rotation'], '0 or 1'),
        (r'$R_{dir,3}$', colors['rotation'], '0 or 1'),
        (r'$B_{id,1}$', colors['beam_exterior'], 'Exterior'),
        (r'$B_{id,2}$', colors['beam_interior'], 'Interior')
    ]

    x_start = 0.05
    y_center = 0.55
    w, h = 0.11, 0.20
    
    # Draw chromosome container (Shadow)
    ax2.add_patch(patches.Rectangle((x_start-0.01, y_center-h/2-0.01), 8*w+0.02, h+0.02, fc='#f0f0f0', ec='none', zorder=0))

    for i, (symbol, color, note) in enumerate(gene_struct):
        # Gene block
        rect = patches.Rectangle((x_start + i*w, y_center - h/2), w, h, 
                                 fc=color, ec='white', lw=2, alpha=0.9, zorder=2)
        ax2.add_patch(rect)
        
        # Symbol
        ax2.text(x_start + i*w + w/2, y_center, symbol, ha='center', va='center', 
                 color='white' if i < 3 or i >= 6 else 'black', fontsize=14, fontweight='bold')
        
        # Annotation
        ax2.text(x_start + i*w + w/2, y_center - h/2 - 0.05, note, ha='center', va='top', fontsize=10, fontstyle='italic')

    # Mathematical Braces (Top labels)
    def draw_bracket(start_idx, end_idx, text, y, color):
        x1 = x_start + start_idx * w
        x2 = x_start + (end_idx + 1) * w
        ax2.annotate('', xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle='-[, widthB=%.1f, lengthB=0.5'%( (x2-x1)*12 ), color=color, lw=2))
        ax2.text((x1+x2)/2, y + 0.08, text, ha='center', va='center', fontsize=12, fontweight='bold', color=color)

    draw_bracket(0, 2, "Section Indices", y_center + h/2 + 0.02, colors['col_corner'])
    draw_bracket(3, 5, "Binary Rotation", y_center + h/2 + 0.02, colors['col_interior'])
    draw_bracket(6, 7, "Beam Sections", y_center + h/2 + 0.02, colors['beam_exterior'])

    # Central Mapping Logic Text
    fig.text(0.5, 0.18, r"$\mathbf{X} = [ \{C_{id}\}_{i=1 \dots 3}, \{R_{dir}\}_{i=1 \dots 3}, \{B_{id}\}_{j=1 \dots 2} ]^T$", 
             ha='center', fontsize=18, color='#34495E',
             bbox=dict(boxstyle='round,pad=0.8', fc='white', ec='#BDC3C7', lw=1.5, alpha=0.9))

    # Legend for Column Types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors['col_corner'], lw=4, label='Corner Column Group'),
        Line2D([0], [0], color=colors['col_edge'], lw=4, label='Edge Column Group'),
        Line2D([0], [0], color=colors['col_interior'], lw=4, label='Interior Column Group'),
        Line2D([0], [0], color=colors['beam_exterior'], lw=4, label='Exterior Beam Group'),
        Line2D([0], [0], color=colors['beam_interior'], lw=4, label='Interior Beam Group')
    ]
    ax2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.45), 
               ncol=2, frameon=False, fontsize=11)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_axis_off()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    
    # Save with high quality
    output_path = 'SCI/Figure1_Mapping.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Professional Figure 1 saved to {output_path}")

if __name__ == "__main__":
    draw_figure1_professional()
