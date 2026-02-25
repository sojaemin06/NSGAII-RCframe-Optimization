import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- SCI Paper Style Configuration ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

def draw_pro_box(ax, x, y, width, height, text, color='#F0F0F0', header=None, alpha=0.9):
    # Draw Shadow
    shadow_offset = 0.05
    ax.add_patch(patches.Rectangle((x + shadow_offset, y - shadow_offset), width, height, 
                                   fc='gray', alpha=0.2, zorder=0, lw=0))
    
    # Draw Main Box
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='#34495E', 
                             facecolor=color, alpha=alpha, zorder=1)
    ax.add_patch(rect)
    
    # Text
    v_offset = 0.15 if header else 0
    if header:
        ax.text(x + width/2, y + height - 0.2, header, ha='center', va='center', 
                fontweight='bold', fontsize=12, color='#2C3E50', zorder=2)
        
    ax.text(x + width/2, y + height/2 - v_offset, text, ha='center', va='center', 
            fontweight='normal', wrap=True, zorder=2)

def draw_arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(label if label else '', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='#2C3E50', lw=1.5, mutation_scale=20),
                ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=3)

def draw_figure4_professional():
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')

    # Professional Palette
    colors = {
        'input': '#EBF5FB',      # Light Blue
        'column': '#D6EAF8',     # Navy Tint
        'beam': '#FDEDEC',       # Red Tint
        'eval': '#FADBD8',       # Muted Red
        'reduction': '#FEF9E7',  # Soft Yellow
        'output': '#EAECEE'      # Slate Gray
    }

    # --- 1. Input Section ---
    draw_pro_box(ax, 3.5, 10.5, 5, 0.8, "User-Defined Range & Architecture Constraints\n($b, h, f_{ck}, f_y, L_{max}$)", 
                 color=colors['input'], header="[INPUT PARAMETERS]")
    
    draw_arrow(ax, 6, 10.5, 6, 9.8)
    draw_pro_box(ax, 3.5, 9.0, 5, 0.8, "Integrated Candidate Generation\n(Random Permutation & Sampling)", 
                 color=colors['input'])

    # Branching
    draw_arrow(ax, 5.5, 9.0, 3, 8.3) 
    draw_arrow(ax, 6.5, 9.0, 9, 8.3)

    # --- 2. Process Containers (Backgrounds) ---
    # Column Container
    ax.add_patch(patches.Rectangle((0.3, 4.0), 5.4, 4.8, fc='#F4F6F7', ec='#BDC3C7', ls='--', lw=1, zorder=0))
    ax.text(3, 8.5, "COLUMN DETAILING LOGIC", ha='center', fontweight='bold', color='#2E86C1')

    # Beam Container
    ax.add_patch(patches.Rectangle((6.3, 4.0), 5.4, 4.8, fc='#FDF2E9', ec='#EDBB99', ls='--', lw=1, zorder=0))
    ax.text(9, 8.5, "BEAM DETAILING LOGIC", ha='center', fontweight='bold', color='#CA6F1E')

    # --- Column Steps ---
    draw_pro_box(ax, 1, 7.3, 4, 0.7, "Step C1: Place Corner Reinforcements\n(Base 4-bars Selection)", color=colors['column'])
    draw_arrow(ax, 3, 7.3, 3, 6.7)
    draw_pro_box(ax, 1, 6.0, 4, 0.7, "Step C2: Distribute Side Bars\n(Symmetry & Ratio Check)", color=colors['column'])
    draw_arrow(ax, 3, 6.0, 3, 5.4)
    draw_pro_box(ax, 1, 4.7, 4, 0.7, "Step C3: ACI 318 Confinement\n(Supplemental Ties & Spacing)", color=colors['column'])

    # --- Beam Steps ---
    draw_pro_box(ax, 7, 7.3, 4, 0.7, "Step B1: Rebar Layering & Layout\n(Tensile/Compressive Zones)", color=colors['beam'])
    draw_arrow(ax, 9, 7.3, 9, 6.7)
    draw_pro_box(ax, 7, 6.0, 4, 0.7, r"Step B2: Effective Depth ($d_{actual}$)" + "\n(Based on Precise Clearance)", color=colors['beam'])
    draw_arrow(ax, 9, 6.0, 9, 5.4)
    draw_pro_box(ax, 7, 4.7, 4, 0.7, "Step B3: Shear Reinforcement\n(Stirrup Pattern & Length)", color=colors['beam'])

    # --- 3. Evaluation ---
    draw_arrow(ax, 3, 4.7, 5.5, 4.1)
    draw_arrow(ax, 9, 4.7, 6.5, 4.1)
    
    draw_pro_box(ax, 3.5, 3.0, 5, 1.1, 
                 "Structural Performance (P-M, $M_n, V_n$)\n" + 
                 "Material Quantities ($V_{conc}, W_{steel}$)\n" +
                 "Cost & $CO_2$ Emission Mapping", 
                 color=colors['eval'], header="INTEGRATED EVALUATION")

    # --- 4. Strategic Reduction (Filter Shape) ---
    draw_arrow(ax, 6, 3.0, 6, 2.3)
    
    # Draw a Trapezoid for Filter
    filter_poly = patches.Polygon([[3.5, 2.2], [8.5, 2.2], [7.5, 1.0], [4.5, 1.0]], 
                                  closed=True, fc=colors['reduction'], ec='#34495E', lw=2, zorder=1)
    ax.add_patch(filter_poly)
    ax.text(6, 1.6, "STRATEGIC REDUCTION\n(Pareto-Optimal Selection per Group)", 
            ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Legend for Pareto filter
    ax.text(6, 0.8, "Filtering dominated sections within $(b, h)$ groups", 
            ha='center', va='center', fontsize=9, fontstyle='italic')

    # --- 5. Output ---
    draw_arrow(ax, 6, 1.0, 6, 0.5)
    draw_pro_box(ax, 3.5, -0.3, 5, 0.8, 
                 "Project-Specific Section Catalog\n(Ready for GA Optimization)", 
                 color=colors['output'], header="[FINAL DATABASE]")

    # Overall Label
    ax.text(6, 11.8, "Figure 4. Flowchart of the Structural Section Detailing and Pareto-based Reduction Process", 
            ha='center', fontsize=15, fontweight='bold', color='#2C3E50')

    plt.tight_layout()
    output_path = 'SCI/Figure4_DB_Workflow.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Professional Figure 4 saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    draw_figure4_professional()
