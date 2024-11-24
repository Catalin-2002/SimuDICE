import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def parse_file(file_name):
    max_episodes_re = re.compile(r"Running SimuDICE for (\d+) episodes")
    
    simudice_re = re.compile(
        r"SimuDICE sampling strategy: ([1-3])\nAverage reward: ([\-0-9\.]+), with min reward ([\-0-9\.]+) and max reward ([\-0-9\.]+)"
    )
    
    with open(file_name, "r") as file:
        content = file.read()
    
    max_episodes = [int(e) for e in max_episodes_re.findall(content)]
    
    blocks = re.split(r"Running SimuDICE for \d+ episodes", content)
    blocks = blocks[1:]
    
    episodes_count = len(max_episodes)
    sim_steps = ["1", "2", "3"] 
    
    assert len(blocks) == episodes_count, "Mismatch in number of episode blocks."
    
    data = {
        "Formula 1": {"avg": [], "min": [], "max": []},
        "Formula 2": {"avg": [], "min": [], "max": []},
        "Formula 3": {"avg": [], "min": [], "max": []},
    }
    
    for block_index, block in enumerate(blocks):
        matches = simudice_re.findall(block)
        
        assert len(matches) == len(sim_steps), f"Mismatch in number of Formula entries within block {block_index + 1}."
        
        for match in matches:
            ss, avg, mn, mx = match
            key = f"Formula {ss}"
            data[key]["avg"].append(float(avg))
            data[key]["min"].append(float(mn))
            data[key]["max"].append(float(mx))
    data["max_episodes"] = max_episodes
    
    return data

def plot_files(file_matrix, env_names):
    num_envs = len(file_matrix)
    num_epsilons = len(file_matrix[0])
    
    ss_colors = {
        "Formula 1": "red",
        "Formula 2": "blue",
        "Formula 3": "orange"
    }
    
    ss_labels = ["Formula 1", "Formula 2", "Formula 3"]
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
            for ss in ss_labels:
                all_min.append(min(data[ss]["min"]))
                all_max.append(max(data[ss]["max"]))
        
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
            
            for ss in ss_labels:
                avg = data[ss]["avg"]
                mn = data[ss]["min"]
                mx = data[ss]["max"]
                
                ax.plot(max_episodes, avg, label=ss, color=ss_colors[ss])
                ax.fill_between(max_episodes, mn, mx, color=ss_colors[ss], alpha=0.2)
            
            ax.set_title(f'{env_names[i]}, ε = {0.1 + 0.3 * j:.1f}', fontsize=12, fontweight='bold')
            
            if j == 0:
                ax.set_ylabel('Average Per-Step Reward', fontsize=10)
            if i == num_envs - 1:
                ax.set_xlabel('Number of Episodes', fontsize=10)
            
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_ylim(y_limits[i])
            
            ax.legend().set_visible(False)
            
            ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))  
            ax.yaxis.set_major_locator(plt.MaxNLocator(4)) 
    
    handles = []
    labels = []
    for ss in ss_labels:
        handles.append(plt.Line2D([], [], color=ss_colors[ss], label=ss))
        labels.append(ss)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.2)
    
    plt.tight_layout()
    
    filename = 'different_formulas_comparison'
    plt.savefig(f"{filename}.png", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
    plt.show()

file_matrix = [
    ['taxi_0.1_different_formulas', 'taxi_0.4_different_formulas', 'taxi_0.7_different_formulas']
]

file_matrix = [
    [f"{file_name}.txt" for file_name in file_names] for file_names in file_matrix
]
env_names = ["Taxi"]  

plot_files(file_matrix, env_names)
