import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def parse_file(file_name):
    max_episodes_re = re.compile(r"Running SimuDICE for (\d+) episodes")

    q_learning_re = re.compile(
        r"Q-learning\s+Average reward: ([\-0-9\.]+), with min reward ([\-0-9\.]+) and max reward ([\-0-9\.]+)"
    )
    offline_dyna_q_10_re = re.compile(
        r"Offline Dyna-Q 10 PS\s+Average reward: ([\-0-9\.]+), with min reward ([\-0-9\.]+) and max reward ([\-0-9\.]+)"
    )
    offline_dyna_q_20_re = re.compile(
        r"Offline Dyna-Q 20 PS\s+Average reward: ([\-0-9\.]+), with min reward ([\-0-9\.]+) and max reward ([\-0-9\.]+)"
    )
    simudice_10_re = re.compile(
        r"SimuDICE 10 PS\s+Average reward: ([\-0-9\.]+), with min reward ([\-0-9\.]+) and max reward ([\-0-9\.]+)"
    )
    simudice_20_re = re.compile(
        r"SimuDICE 20 PS\s+Average reward: ([\-0-9\.]+), with min reward ([\-0-9\.]+) and max reward ([\-0-9\.]+)"
    )

    max_episodes = []
    q_learning_avg = []
    q_learning_min = []
    q_learning_max = []
    offline_dyna_q_10_avg = []
    offline_dyna_q_10_min = []
    offline_dyna_q_10_max = []
    offline_dyna_q_20_avg = []
    offline_dyna_q_20_min = []
    offline_dyna_q_20_max = []
    simudice_10_avg = []
    simudice_10_min = []
    simudice_10_max = []
    simudice_20_avg = []
    simudice_20_min = []
    simudice_20_max = []

    with open(file_name, "r") as file:
        content = file.read()

    max_episodes = [int(e) for e in max_episodes_re.findall(content)]
    q_matches = q_learning_re.findall(content)
    dyna_q_10_matches = offline_dyna_q_10_re.findall(content)
    dyna_q_20_matches = offline_dyna_q_20_re.findall(content)
    simudice_10_matches = simudice_10_re.findall(content)
    simudice_20_matches = simudice_20_re.findall(content)

    num_runs = len(max_episodes)
    assert all(len(lst) == num_runs for lst in [
        q_matches, dyna_q_10_matches, dyna_q_20_matches,
        simudice_10_matches, simudice_20_matches
    ]), "Mismatch in number of runs and algorithm entries."

    for match in q_matches:
        avg, mn, mx = match
        q_learning_avg.append(float(avg))
        q_learning_min.append(float(mn))
        q_learning_max.append(float(mx))

    for match in dyna_q_10_matches:
        avg, mn, mx = match
        offline_dyna_q_10_avg.append(float(avg))
        offline_dyna_q_10_min.append(float(mn))
        offline_dyna_q_10_max.append(float(mx))

    for match in dyna_q_20_matches:
        avg, mn, mx = match
        offline_dyna_q_20_avg.append(float(avg))
        offline_dyna_q_20_min.append(float(mn))
        offline_dyna_q_20_max.append(float(mx))

    for match in simudice_10_matches:
        avg, mn, mx = match
        simudice_10_avg.append(float(avg))
        simudice_10_min.append(float(mn))
        simudice_10_max.append(float(mx))

    for match in simudice_20_matches:
        avg, mn, mx = match
        simudice_20_avg.append(float(avg))
        simudice_20_min.append(float(mn))
        simudice_20_max.append(float(mx))

    return {
        "max_episodes": max_episodes,
        "Q-learning": {
            "avg": q_learning_avg,
            "min": q_learning_min,
            "max": q_learning_max
        },
        "Offline Dyna-Q 10 PS": {
            "avg": offline_dyna_q_10_avg,
            "min": offline_dyna_q_10_min,
            "max": offline_dyna_q_10_max
        },
        "Offline Dyna-Q 20 PS": {
            "avg": offline_dyna_q_20_avg,
            "min": offline_dyna_q_20_min,
            "max": offline_dyna_q_20_max
        },
        "SimuDICE 10 PS": {
            "avg": simudice_10_avg,
            "min": simudice_10_min,
            "max": simudice_10_max
        },
        "SimuDICE 20 PS": {
            "avg": simudice_20_avg,
            "min": simudice_20_min,
            "max": simudice_20_max
        }
    }

def plot_files(file_matrix, env_names):
    num_envs = len(file_matrix)
    num_epsilons = len(file_matrix[0])

    algorithm_colors = {
        "Q-learning": "green",
        "Offline Dyna-Q 10 PS": "orange",
        "Offline Dyna-Q 20 PS": "purple",
        "SimuDICE 10 PS": "blue",
        "SimuDICE 20 PS": "red"
    }

    algorithms = ["Q-learning", "Offline Dyna-Q 10 PS", "Offline Dyna-Q 20 PS",
                  "SimuDICE 10 PS", "SimuDICE 20 PS"]

    total_plots = num_envs * num_epsilons
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 8 * total_plots), sharex=False, sharey=False)

    if total_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    y_limits = {}

    for i in range(num_envs):
        for j in range(num_epsilons):
            file_name = file_matrix[i][j]
            data = parse_file(file_name)
            all_min = []
            all_max = []
            for algo in algorithms:
                all_min.append(min(data[algo]["min"]))
                all_max.append(max(data[algo]["max"]))
            
            env = env_names[i].lower()
            if env == "taxi":
                y_min = -6.0
                y_max = 1.0

            y_limits[(i, j)] = (y_min, y_max)

    plot_idx = 0 
    for i in range(num_envs):
        for j in range(num_epsilons):
            file_name = file_matrix[i][j]
            data = parse_file(file_name)

            ax = axes[plot_idx]

            max_episodes = data["max_episodes"]

            for algo in algorithms:
                avg = data[algo]["avg"]
                mn = data[algo]["min"]
                mx = data[algo]["max"]

                ax.plot(max_episodes, avg, label=algo, color=algorithm_colors[algo], linewidth=2)
                
                if env_names[i].lower() == "cliff walking":
                    ax.fill_between(max_episodes, mn, mx, color=algorithm_colors[algo], alpha=0.1)
                else:
                    ax.fill_between(max_episodes, mn, mx, color=algorithm_colors[algo], alpha=0.2)

            ax.set_title(f'{env_names[i]}, ε = {0.1 + 0.3 * j:.1f}', fontsize=16, fontweight='bold')

            ax.set_ylabel('Average Cumulative Reward', fontsize=14)

            if plot_idx == total_plots - 1:
                ax.set_xlabel('Number of Episodes', fontsize=14)

            ax.tick_params(axis='both', which='major', labelsize=12)

            ax.set_ylim(y_limits[(i, j)])

            ax.legend().set_visible(False)

            ax.grid(False, which='both', linestyle='--', linewidth=0.3, alpha=0.5)

            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))

            plot_idx += 1 

    handles = []
    labels = []
    for algo, color in algorithm_colors.items():
        handles.append(plt.Line2D([], [], color=color, label=algo, linewidth=2))
        labels.append(algo)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=14)

    plt.subplots_adjust(wspace=0.1, hspace=0.4, top=0.95, bottom=0.1)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    filename = 'poster_plot'
    plt.savefig(f"{filename}.png", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.svg", format='svg', bbox_inches='tight')

    plt.show()

file_matrix = [
    ['taxi_0.1_overall', 'taxi_0.4_overall', 'taxi_0.7_overall']
]

file_matrix = [
    [f"{file_name}.txt" for file_name in file_names] for file_names in file_matrix
]
env_names = ["Taxi"]

plot_files(file_matrix, env_names)
