import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_IF_fig2(chosen_test_examples, folder_influence, model_name, formatname = 'png', name = "IF"):
    # Ready to use

    # Overriding fonts
    plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
    })
    plt.rc('text', usetex=True)

    # Loading the mask
    mask = np.load('model/' + model_name + '_mask.npy') 
    antimask = np.argsort(mask)

    # Training points
    U_array = np.concatenate((np.linspace(0, 1, 500), np.linspace(1.01, 40, 500)))
    
    U_testarray = np.concatenate((np.linspace(0.01, 0.999, 20), np.linspace(1.02, 2, 5), np.linspace(2.066, 39, 20)))
    max_x = 40
    min_y = -1e-2
    max_y = 1e-1
    trans_point = 1
    U_value = '0'
    xticks_location = np.concatenate((np.array([0, 1, 2]), np.array([4,10,40])))
    xticks_labels = np.concatenate((np.array(['0', '1', '2']), np.array(['4','10','40'])))
    yticks_location = np.array([-1e-3, 0, 1e-3, 1e-1])

    # Seaborn style set
    sns.set(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': 'dashed', "grid.color": "0.6", 'axes.edgecolor': '.1'})

    # Plot colors
    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    palette_background = sns.light_palette((210, 90, 60), input="husl")
    palette_background = sns.light_palette("lightsteelblue", 6)
    c_left = sns.light_palette("navy")[-2]
    c_right = sns.light_palette("purple")[-2]
    c_help = sns.light_palette("green")[-2]
    c_harm = sns.light_palette("red")[-2]
    c_phase1 = palette_background[0]
    c_phase2 = palette_background[2]
    c_test = sns.xkcd_palette(colors)[1]

    marker_size = 0.5
    marker_size_help = 1.5

    i = 0
    j = 0
    fig, axs = plt.subplots(2, 2, figsize=(3 + 3/8, 2.5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.rc('font', size=9)
    plt.rc('axes', labelsize=8)

    for test_sample in chosen_test_examples:

        # Influence functions of all train elements for one test example
        with open(folder_influence + '/original_influence_test' + str(test_sample) + '.txt') as filelabels:
            influence_functions = np.loadtxt(filelabels, dtype=float)

            antimasked_inf_funs = influence_functions[antimask]
            sorting_indices = np.argsort(antimasked_inf_funs)

        antimasked_inf_funs_phase1 = antimasked_inf_funs[0:502]
        antimasked_inf_funs_phase2 = antimasked_inf_funs[502:1001]
        U_array_phase1 = U_array[0:502]
        U_array_phase2 = U_array[502:1001]

        U_test_value = U_testarray[test_sample]

        # Figure
        axs[i][j].scatter(U_array_phase1, antimasked_inf_funs_phase1, marker='o', c=c_left, s=marker_size, label='training points, phase 1')
        axs[i][j].scatter(U_array_phase2, antimasked_inf_funs_phase2, marker='o', c=c_right, s=marker_size, label='training points, phase 2')
        axs[i][j].plot(U_array[sorting_indices[:5]], antimasked_inf_funs[sorting_indices[:5]], 'o', c=c_harm, markersize=marker_size_help, label='most harmful')
        axs[i][j].plot(U_array[sorting_indices[-5:]], antimasked_inf_funs[sorting_indices[-5:]], 'o', c=c_help, markersize=marker_size_help, label='most helpful')
        axs[i][j].plot([U_test_value,U_test_value], [min_y, max_y], color=c_test, label='test point (U\'\'=' + U_value + ')')

        axs[i][j].set_yscale('symlog', linthreshy=1e-3)
        axs[i][j].set_xscale('symlog', linthreshx=3)

        axs[i][j].set_xticks(xticks_location)
        axs[i][j].set_xticklabels(xticks_labels)
        axs[i][j].set_yticks(yticks_location)
        axs[i][j].tick_params(which='both', labelsize='small')

        axs[i][j].set_ylim(min_y, max_y)
        axs[i][j].set_xlim(0, max_x)
        axs[i][j].tick_params(which='both', direction='in')

        axs[i][j].grid(linewidth=0.1)

        for axis in ['top','bottom','left','right']:
            axs[i][j].spines[axis].set_linewidth(0.05)

        # Two colors background
        axs[i][j].axvspan(0, trans_point, facecolor=c_phase1, zorder=0, lw=0)
        axs[i][j].axvspan(trans_point, max_x, facecolor=c_phase2, zorder=0, lw=0)

        j += 1
        if (j % 2 == 0):
            i += 1
            j = 0

    axs[0][0].text(35, 0.06, '(a)', verticalalignment='top', horizontalalignment='right', family="serif")
    axs[0][1].text(0.15, 0.06, '(b)', verticalalignment='top', horizontalalignment='left', family="serif")
    axs[1][0].text(35, 0.06, '(c)', verticalalignment='top', horizontalalignment='right', family="serif")
    axs[1][1].text(0.15, 0.06, '(d)', verticalalignment='top', horizontalalignment='left', family="serif")
    
    IFtext = fig.text(-0.03, 0.5, 'Influence function value', family="serif", va='center', rotation='vertical')
    Utext = fig.text(0.5, -0.01, '$V_1/\,J$ ', family="serif", ha='center')

    #fig.savefig('./figures/' + name + '.' + formatname, bbox_extra_artists=(Utext, IFtext), bbox_inches='tight')
    return fig