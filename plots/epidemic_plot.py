import json
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 25

model_types = ['SIS', 'SIR', 'SEIR']
data = []
ROOT_PATH = os.path.dirname(os.path.abspath(__file__)).split("DINO")[0] + "DINO/"

files = os.listdir(f'{ROOT_PATH}saved_results/')
for model in model_types:
    file = [f for f in files if model in f]
    if len(file) != 0:
        data.append(json.load(open(f'{ROOT_PATH}saved_results/{file[0]}', "r")))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))
ax_list = [ax1, ax2, ax3]
methods = ['random', 'degree_direct', 'katz_centrality', 'closeness_centrality', 'hits','page_rank', 'acquaintance', 'cbf', 'dino']
labels = ['Random', 'Degree', 'Katz.', 'Close.', 'HITS', 'PageRank', 'Acqu.', 'CBF', 'DINO']
colors = ["#CB9AF3","#7763D3", "#0C3E95", "#3288bd", "#abdda4", "#028A46","#FF983B", "#F8AFF6", "#d53e4f"]
markers = ['o', 's', 'H', '^', 'v', '<', '>', 'p',  'D']

for a in range(len(data)):
    for i in range(len(labels)):
        if labels[i] == "DINO":
            ax_list[a].plot(data[a][methods[i]]['t'], data[a][methods[i]]['i'], label=labels[i], marker=markers[i], markersize=10, color=colors[i],markeredgecolor="black", markeredgewidth=2)
        else:
            ax_list[a].plot(data[a][methods[i]]['t'], data[a][methods[i]]['i'], label=labels[i], marker=markers[i], markersize=10, color=colors[i])
        ax_list[a].set_xlabel("Time", fontname='Times New Roman', fontsize=25)
        ax_list[a].set_facecolor("#eeeeee")
        ax_list[a].grid(True, which='both', linestyle='--', linewidth=0.5, color="grey")

plt.tight_layout()
legend = ax2.legend(loc='upper center', bbox_to_anchor=(0.5,1.2),
          fancybox=False, shadow=False,frameon=False, ncol=5) 
for handle in legend.legend_handles:
    handle.set_markersize(20)
ax1.text(0.5, -0.13, '(a) SIS ($I_0$ = 0.95, $\\beta$=0.03)', ha='center', va='top', transform=ax1.transAxes, fontsize=25)
ax2.text(0.5, -0.13, '(b) SIR ($I_0$ = 0.95, $\\beta$=0.03, $\\alpha_{IR}$ = 0.8)', ha='center', va='top', transform=ax2.transAxes, fontsize=25)
ax3.text(0.5, -0.13, '(c) SEIR ($I_0$ = 0.95, $\\beta_1$=0.00,$\\beta_2$=0.03, \n $\\alpha_{EI}$ = 0.1, $\\alpha_{IR}$ = 0.8)', ha='center', va='top', transform=ax3.transAxes, fontsize=25)
ax1.set_ylabel("Infected people", fontname='Times New Roman', fontsize=25)
plt.savefig("fig3.pdf",bbox_inches="tight")
plt.show()