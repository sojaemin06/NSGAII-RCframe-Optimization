import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- SCI Paper Style Configuration ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

def draw_box(ax, x, y, width, height, text, color='#F0F0F0', boxstyle="round,pad=0.1"):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle=boxstyle,
                                  linewidth=1.2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

fig, ax = plt.subplots(figsize=(11, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 11)
ax.axis('off')

# --- 1. Common Input ---
draw_box(ax, 4, 10, 4, 0.7, "Input Parameters\n(Range of b, h, fck, fy)", color='#D1E8FF')
draw_arrow(ax, 6, 10, 6, 9.5)
draw_box(ax, 4, 8.8, 4, 0.7, "Random Sampling Combination\n(Initial Pool Generation)", color='#E2F0D9')

# --- Branching ---
draw_arrow(ax, 6, 8.8, 3, 8.2) # To Column
draw_arrow(ax, 6, 8.8, 9, 8.2) # To Beam

# --- 2. Column Detailing Logic ---
draw_box(ax, 1, 7.5, 4, 0.7, "Step C1: Place Corner Bars\n(Fixed 4-bars)", color='#FFF2CC')
draw_arrow(ax, 3, 7.5, 3, 6.8)
draw_box(ax, 1, 6.1, 4, 0.7, "Step C2: Distribute Side Bars\n(Front-Back & Left-Right Pairs)", color='#FFF2CC')
draw_arrow(ax, 3, 6.1, 3, 5.4)
draw_box(ax, 1, 4.7, 4, 0.7, "Step C3: Center Spacing Check\n(Max 150mm for Supplemental Ties)", color='#FFF2CC')

# --- 3. Beam Detailing Logic ---
draw_box(ax, 7, 7.5, 4, 0.7, "Step B1: Flexural Rebar Layering\n(Tensile & Compressive Layers)", color='#FFF2CC')
draw_arrow(ax, 9, 7.5, 9, 6.8)
draw_box(ax, 7, 6.1, 4, 0.7, "Step B2: Calculate d_actual\n(Based on N_layers & Bar Size)", color='#FFF2CC')
draw_arrow(ax, 9, 6.1, 9, 5.4)
draw_box(ax, 7, 4.7, 4, 0.7, "Step B3: Shear Rebar Design\n(Stirrup Size & Spacing)", color='#FFF2CC')

# --- 4. Strength & Eco Evaluation (Unified) ---
draw_arrow(ax, 3, 4.7, 5.5, 4.0)
draw_arrow(ax, 9, 4.7, 6.5, 4.0)
draw_box(ax, 4, 3.2, 4, 0.8, "Evaluation Module\n- Strength (P-M, Mn, Vn)\n- Economic (Cost)\n- Env. (CO2 Emission)", color='#F8CECC')

# --- 5. Strategic Reduction ---
draw_arrow(ax, 6, 3.2, 6, 2.4)
draw_box(ax, 3.5, 1.4, 5, 1.0, "Strategic Reduction Logic\n(Group by b,h and Select\nPareto-Optimal within Group)", color='#FFD966')

# --- 6. Final Output ---
draw_arrow(ax, 6, 1.4, 6, 0.7)
draw_box(ax, 4, 0, 4, 0.7, "Final Project Catalog\n(800 Columns, 500 Beams)", color='#D9D9D9')

# Annotations
ax.text(3, 10.8, "Database Detailing & Reduction Algorithm", ha='center', fontsize=12, fontweight='bold', color='blue')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SCI', 'Figure4_DB_Workflow.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Workflow Diagram saved to {output_path}")
plt.close()
