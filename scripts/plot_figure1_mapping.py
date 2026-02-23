import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_figure1():
    fig = plt.figure(figsize=(14, 8))
    
    # --- 1. 3D Frame View (Left) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("1. 3D Frame Member Grouping", fontsize=14, pad=20)
    
    # Simple 1-bay, 2-story frame
    def draw_member(p1, p2, color, lw=3):
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=lw)

    # Coordinates
    nodes = {
        0: (0,0,0), 1: (5,0,0), 2: (5,5,0), 3: (0,5,0),
        4: (0,0,4), 5: (5,0,4), 6: (5,5,4), 7: (0,5,4),
        8: (0,0,8), 9: (5,0,8), 10: (5,5,8), 11: (0,5,8)
    }
    
    # Columns (Grouped by color)
    col_group1 = [(0,4), (1,5), (2,6), (3,7)] # 1st floor
    col_group2 = [(4,8), (5,9), (6,10), (7,11)] # 2nd floor
    
    for start, end in col_group1:
        draw_member(nodes[start], nodes[end], 'royalblue', 5)
    for start, end in col_group2:
        draw_member(nodes[start], nodes[end], 'skyblue', 5)
        
    # Beams (Grouped by color)
    beam_group1 = [(4,5), (5,6), (6,7), (7,4)] # 1st floor beams
    beam_group2 = [(8,9), (9,10), (10,11), (11,8)] # 2nd floor beams
    
    for start, end in beam_group1:
        draw_member(nodes[start], nodes[end], 'tomato', 4)
    for start, end in beam_group2:
        draw_member(nodes[start], nodes[end], 'orange', 4)

    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=45)

    # --- 2. Chromosome Structure (Right) ---
    ax2 = fig.add_subplot(122)
    ax2.set_title("2. Genetic Chromosome Mapping", fontsize=14, pad=20)
    
    # Draw blocks for Chromosome
    # Structure: [C1, C2 | R1, R2 | B1, B2]
    blocks = [
        ('C_Group 1', 'royalblue'), ('C_Group 2', 'skyblue'),
        ('R_Dir 1', 'lightgrey'), ('R_Dir 2', 'lightgrey'),
        ('B_Group 1', 'tomato'), ('B_Group 2', 'orange')
    ]
    
    x_start = 0.1
    y_center = 0.5
    width = 0.12
    height = 0.15
    
    for i, (label, color) in enumerate(blocks):
        rect = patches.Rectangle((x_start + i*width, y_center - height/2), width, height, 
                                 edgecolor='black', facecolor=color, alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(x_start + i*width + width/2, y_center, label, ha='center', va='center', rotation=90, fontweight='bold')
        
        # Add values example
        val = "ID: 45" if 'Group' in label else "0 or 1"
        ax2.text(x_start + i*width + width/2, y_center - height, val, ha='center', va='center', fontsize=9)

    # Draw Curly Braces (Conceptual)
    ax2.annotate('', xy=(x_start, y_center + height), xytext=(x_start + 2*width, y_center + height),
                arrowprops=dict(arrowstyle='<->', color='blue'))
    ax2.text(x_start + width, y_center + height*1.2, "Column Sections", ha='center')
    
    ax2.annotate('', xy=(x_start + 2*width, y_center + height), xytext=(x_start + 4*width, y_center + height),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax2.text(x_start + 3*width, y_center + height*1.2, "Rotations", ha='center')

    ax2.annotate('', xy=(x_start + 4*width, y_center + height), xytext=(x_start + 6*width, y_center + height),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax2.text(x_start + 5*width, y_center + height*1.2, "Beam Sections", ha='center')

    # Mapping arrows from 3D to Gene (Conceptual)
    # This is hard to do precisely across subplots, so we'll add text
    fig.text(0.5, 0.45, "Mapping: Group IDs to Gene Indices", ha='center', fontsize=12, 
             bbox=dict(boxstyle='rarrow', fc='white', ec='black', lw=1))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig('SCI/Figure1_Mapping.png', dpi=300)
    print("Figure 1 saved to SCI/Figure1_Mapping.png")

if __name__ == "__main__":
    draw_figure1()
