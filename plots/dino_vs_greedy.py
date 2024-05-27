from matplotlib import pyplot as plt


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 30

colors=["#d53e4f","#7985A5"]

# Plot data
node_budget = list(range(1,11))
node_budget_greedy = [0.034, 0.067, 0.097, 0.127, 0.156, 0.185, 0.212, 0.239, 0.266, 0.292]
node_budget_dino = [0.034, 0.064, 0.097, 0.125, 0.154, 0.181, 0.210, 0.236, 0.261, 0.291]

markers = ['o', 'D',]
ax1.plot(node_budget, node_budget_greedy, label='Greedy', marker='o', color=colors[1],markersize=15,markeredgecolor="black", markeredgewidth=1)
ax1.plot(node_budget, node_budget_dino, label='DINO', marker='D', color=colors[0], markersize=15,markeredgecolor="black", markeredgewidth=3)
ax1.set_ylim(0,0.42)
ax1.set_xlabel('Node Budget')
ax1.set_ylabel('Spectral Radius Decrease')
ax1.set_title('Approx. w. Various Node Budgets')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
ax1.set_facecolor("#eeeeee")

node_numbers = [300, 500, 800, 1200, 1800]
node_numbers_greedy = [0.52, 0.39, 0.329, 0.254, 0.22]
node_numbers_dino = [0.47, 0.38, 0.324, 0.25, 0.216]

ax2.plot(node_numbers, node_numbers_greedy, label='Greedy', marker='o', color=colors[1],markersize=15,markeredgecolor="black", markeredgewidth=1)
ax2.plot(node_numbers, node_numbers_dino, label='DINO', marker='D', color=colors[0], markersize=15,markeredgecolor="black", markeredgewidth=3)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
ax2.set_facecolor("#eeeeee")
ax2.set_title('Approx. w. Various Node Numbers')
ax2.set_xlabel('Number of Nodes in Network')
ax2.legend()

ax1.text(0.5, -0.13, '(a)', ha='center', va='top', transform=ax1.transAxes, fontsize=30)
ax2.text(0.5, -0.13, '(b)', ha='center', va='top', transform=ax2.transAxes, fontsize=30)
plt.savefig("fig2.pdf",bbox_inches="tight")
plt.show()