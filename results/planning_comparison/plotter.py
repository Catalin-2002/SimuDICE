import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def parse_file(file_name):
    max_episodes_re = re.compile(r"Running SimuDICE for (\d+) episodes")

    simudice_re = re.compile(
        r"SimuDICE PS: (\d+)\nAverage reward: ([-\d\.]+), with min reward ([-\d\.]+) and max reward ([-\d\.]+)"
    )
    
    with open(file_name, "r") as file:
        content = file.read()

    max_episodes = [int(e) for e in max_episodes_re.findall(content)]

    data = {
        "max_episodes": max_episodes,
        "SimuDICE PS 1": {"avg": [], "min": [], "max": []},
        "SimuDICE PS 5": {"avg": [], "min": [], "max": []},
        "SimuDICE PS 10": {"avg": [], "min": [], "max": []},
        "SimuDICE PS 20": {"avg": [], "min": [], "max": []},
        "SimuDICE PS 50": {"avg": [], "min": [], "max": []},
    }

    episodes_count = len(max_episodes)
    ps_steps = ["1", "5", "10", "20", "50"]
    num_ps = len(ps_steps)

    blocks = re.split(r"Running SimuDICE for \d+ episodes", content)
    blocks = blocks[1:]

    assert len(blocks) == episodes_count, "Mismatch in number of episode blocks."

    for block in blocks:
        matches = simudice_re.findall(block)
        assert len(matches) == num_ps, "Mismatch in number of PS entries within a block."
        for match in matches:
            ps, avg, mn, mx = match
            key = f"SimuDICE PS {ps}"
            data[key]["avg"].append(float(avg))
            data[key]["min"].append(float(mn))
            data[key]["max"].append(float(mx))

    return data

def plot_files(file_matrix, env_names):
    num_envs = len(file_matrix)
    num_epsilons = len(file_matrix[0])

    ps_colors = {
        "SimuDICE PS 1": "green",
        "SimuDICE PS 5": "red",
        "SimuDICE PS 10": "blue",
        "SimuDICE PS 20": "orange",
        "SimuDICE PS 50": "purple"
    }

    ps_labels = ["SimuDICE PS 1", "SimuDICE PS 5", "SimuDICE PS 10",
                 "SimuDICE PS 20", "SimuDICE PS 50"]
    fig, axes = plt.subplots(num_envs, num_epsilons, figsize=(10, 4), sharey='row')

    if num_envs == 1 and num_epsilons == 1:
        axes = np.array([[axes]])
    elif num_envs == 1 or num_epsilons == 1:
        axes = axes.reshape(num_envs, num_epsilons)
    else:
        axes = axes

    y_limits = {}

    for i in range(num_envs):
        all_min = []
        all_max = []
        for j in range(num_epsilons):
            file_name = file_matrix[i][j]
            data = parse_file(file_name)
            for ps in ps_labels:
                all_min.append(min(data[ps]["min"]))
                all_max.append(max(data[ps]["max"]))

        if env_names[i].lower() == "taxi":
            y_min = -6.0
            y_max = 1.0

        y_limits[i] = (y_min, y_max)

    for j in range(num_epsilons):
        for i in range(num_envs):
            file_name = file_matrix[i][j]
            data = parse_file(file_name)

            ax = axes[i, j]

            max_episodes = data["max_episodes"]

            for ps in ps_labels:
                avg = data[ps]["avg"]
                mn = data[ps]["min"]
                mx = data[ps]["max"]

                ax.plot(max_episodes, avg, label=ps, color=ps_colors[ps])
                ax.fill_between(max_episodes, mn, mx, color=ps_colors[ps], alpha=0.2)

            ax.set_title(f'{env_names[i]}, ε = {0.1 + 0.3 * j:.1f}', fontsize=12, fontweight='bold')

            if j == 0:
                ax.set_ylabel('Average Per-Step Reward', fontsize=10)
            if i == num_envs - 1:
                ax.set_xlabel('Number of Episodes', fontsize=10)

            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_ylim(y_limits[i])

            ax.legend().set_visible(False)

            ax.grid(False, which='both', linestyle='--', linewidth=0.3, alpha=0.5)

            ax.xaxis.set_major_locator(plt.MaxNLocator(4))  
            ax.yaxis.set_major_locator(plt.MaxNLocator(4)) 

    handles = []
    labels = []
    for ps in ps_labels:
        handles.append(plt.Line2D([], [], color=ps_colors[ps], label=ps))
        labels.append(ps)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=10)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.2)

    plt.tight_layout()
    
    filename = 'different_ps_comparison'
    plt.savefig(f"{filename}.png", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
    plt.show()

file_matrix = [
    ['taxi_0.1_different_planning_steps', 'taxi_0.4_different_planning_steps', 'taxi_0.7_different_planning_steps']
]

file_matrix = [
    [f"{file_name}.txt" for file_name in file_names] for file_names in file_matrix
]
env_names = ["Taxi"]

plot_files(file_matrix, env_names)
