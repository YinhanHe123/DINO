from matplotlib import pyplot as plt


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19,9))
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 30

labels = ["Ran.", "Deg.", "Katz.", "Clo.", "HITS", "P.R.","Acqu.", "CBF.",  "DINO", "Gre."]
spectral_radius_drop = [0.007, 0.53, 0.44, 0.416, 0.017, 0.334, 0.0001, 0.5502, 0.755, 0.890]
running_time = [0.0036, 0.0027, 0.258, 63.13, 1.2215, 0.0107, 0.0028, 0.3321, 2.5907, 60379]

markers = ['o', 's', 'H', '^', 'v', '<', '>', 'p',  'D', '*',]
colors = ["#CB9AF3","#7763D3", "#0C3E95", "#3288bd", "#abdda4", "#028A46","#FF983B", "#F8AFF6", "#d53e4f","#7985A5"]
ax1.set_xlim(-2500,63000)
for i, label in enumerate(labels):
    if label == "DINO":
        ax1.scatter(running_time[i], spectral_radius_drop[i], marker=markers[i], s=800, color=colors[i], label=label,edgecolors='black', linewidths=3)
    else:
        ax1.scatter(running_time[i], spectral_radius_drop[i], marker=markers[i], s=800, color=colors[i], label=label,edgecolors='black')
ax1.plot([running_time[labels.index("DINO")], running_time[labels.index("Gre.")]],[spectral_radius_drop[labels.index("DINO")], spectral_radius_drop[labels.index("Gre.")]], '--', color="black")
ax1.legend(loc='lower right', markerscale=1, fontsize=40, ncol=2, columnspacing=0.,
            handletextpad=0.2, frameon=True, borderpad=0.1)
ax1.set_ylabel("Spectral Radius Decrease")
ax1.set_xlabel("Running Time (s)")
ax1.set_title('Quality Time Trade-off')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
ax1.set_facecolor("#eeeeee")

x_vals = [264, 678, 1996, 4242, 9213, 16429, 24643, 31764, 38000, 50491, 60498, 70630, 81395, 92465, 101544]
y_vals = [0.08232426643371582, 0.10591363906860352, 0.1337752342224121, 0.3432091474533081, 0.745266318321228, 1.195417046546936, 1.7065284252166748, 2.408107042312622, 3.1415103673934937, 4.015967965126038, 5.067585587501526, 6.067762732505798, 7.336578369140625, 8.789085745811462, 9.759629845619202]
y_errors = [0.04958295822143555, 0.0009667873382568359, 0.009415864944458008, 0.09038817882537842, 0.14035570621490479, 0.0012153387069702148, 0.026038408279418945, 0.009243249893188477, 0.01626718044281006, 0.009295821189880371, 0.000841975212097168, 0.03262174129486084, 0.00015592575073242188, 0.05888330936431885, 0.060507893562316895]
upper_bound = [0.13190722465515137, 0.10688042640686035, 0.14319109916687012, 0.4335973262786865, 0.8856220245361328, 1.1966323852539062, 1.7325668334960938, 2.4173502922058105, 3.1577775478363037, 4.025263786315918, 5.068427562713623, 6.100384473800659, 7.336734294891357, 8.847969055175781, 9.820137739181519]
lower_bound= [0.03274130821228027, 0.10494685173034668, 0.1243593692779541, 0.2528209686279297, 0.6049106121063232, 1.1942017078399658, 1.6804900169372559, 2.3988637924194336, 3.1252431869506836, 4.006672143936157, 5.066743612289429, 6.0351409912109375, 7.336422443389893, 8.730202436447144, 9.699121952056885]
marker = "D"
ax2.plot(x_vals, y_vals, marker=marker,label="DINO", color="#d53e4f", markersize=15)
# Adding the shaded region here
ax2.fill_between(x_vals, lower_bound, upper_bound, color="#d53e4f", alpha=0.3)
ax2.set_xlabel("|V|+|E|")
ax2.set_ylabel("Running Time")
ax2.legend(loc="upper left")
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey') # To remove the background of the legend
ax2.set_title("Scalability Analysis")
ax2.set_facecolor("#eeeeee")

ax1.text(0.5, -0.18, '(a) in p2p-Gnutella', ha='center', va='top', transform=ax1.transAxes, fontsize=30)
ax2.text(0.5, -0.18, '(b) in Erdős–Rényi', ha='center', va='top', transform=ax2.transAxes, fontsize=30)
plt.savefig("fig4.pdf",bbox_inches="tight")
plt.show()