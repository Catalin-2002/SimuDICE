import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def parse_file(file_name):
    max_episodes_re = re.compile(r"Running SimuDICE for (\d+) episodes")
    
    simudice_re = re.compile(
        r"SimuDICE iterations: (\d+)\nAverage reward: ([\-0-9\.]+), with min reward ([\-0-9\.]+) and max reward ([\-0-9\.]+)"
    )
    
    with open(file_name, "r") as file:
        content = file.read()
    
    max_episodes = [int(e) for e in max_episodes_re.findall(content)]
    
    blocks = re.split(r"Running SimuDICE for \d+ episodes", content)
    blocks = blocks[1:] 
    
    episodes_count = len(max_episodes)
    sim_steps = ["1", "2", "3", "4", "5"]  
    
    if len(blocks) != episodes_count:
        print(f"Warning: Number of blocks ({len(blocks)}) does not match number of episodes ({episodes_count}).")
    
    data = {
        "max_episodes": max_episodes,
    }
    
    for it_num in sim_steps:
        key = f"SimuDICE it {it_num}"
        data[key] = {"avg": [], "min": [], "max": []}
    
    for block_idx, block in enumerate(blocks):
        matches = simudice_re.findall(block)
        if len(matches) != len(sim_steps):
            print(f"Warning: Block {block_idx+1} has {len(matches)} it, expected {len(sim_steps)}.")
        
        for match in matches:
            iter_num, avg, mn, mx = match
            if iter_num not in sim_steps:
                continue  
            key = f"SimuDICE it {iter_num}"
            data[key]["avg"].append(float(avg))
            data[key]["min"].append(float(mn))
            data[key]["max"].append(float(mx))
    
    return data

def plot_files(file_matrix, env_names):
    num_envs = len(file_matrix)
    num_epsilons = len(file_matrix[0])
    
    iteration_colors = {
        "SimuDICE it 1": "red",
        "SimuDICE it 2": "blue",
        "SimuDICE it 3": "green",
        "SimuDICE it 4": "orange",
        "SimuDICE it 5": "purple"
    }
    
    iteration_labels = [
        "SimuDICE it 1",
        "SimuDICE it 2",
        "SimuDICE it 3",
        "SimuDICE it 4",
        "SimuDICE it 5"
    ]
    
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
            for iter_label in iteration_labels:
                if iter_label in data:
                    all_min.append(min(data[iter_label]["min"]))
                    all_max.append(max(data[iter_label]["max"]))
        
        if not all_min or not all_max:
            print(f"Warning: No data found for environment {env_names[i]}.")
            continue
        
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
            
            for iter_label in iteration_labels:
                if iter_label not in data:
                    continue 
                avg = data[iter_label]["avg"]
                mn = data[iter_label]["min"]
                mx = data[iter_label]["max"]
                
                ax.plot(max_episodes, avg, label=iter_label, color=iteration_colors[iter_label])
                ax.fill_between(max_episodes, mn, mx, color=iteration_colors[iter_label], alpha=0.1)
            
            ax.set_title(f'{env_names[i]}, ε = {0.1 + 0.3 * j:.1f}', fontsize=12, fontweight='bold')
            
            if j == 0:
                ax.set_ylabel('Average Per-Step Reward', fontsize=10)
            if i == num_envs - 1:
                ax.set_xlabel('Number of Episodes', fontsize=10)
            
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i in y_limits:
                ax.set_ylim(y_limits[i])
            
            ax.legend().set_visible(False)
            
            ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
        
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))  
            ax.yaxis.set_major_locator(plt.MaxNLocator(4)) 
    
    handles = []
    labels = []
    for iter_label in iteration_labels:
        handles.append(plt.Line2D([], [], color=iteration_colors[iter_label], label=iter_label))
        labels.append(iter_label)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=10)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.2)
    
    plt.tight_layout()
    
    filename = 'different_iteration_no_comparison'
    plt.savefig(f"{filename}.png", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')    
    plt.show()

file_matrix = [
    ['taxi_0.1_different_iteration_no', 'taxi_0.4_different_iteration_no', 'taxi_0.7_different_iteration_no']
]

file_matrix = [
    [f"{file_name}.txt" for file_name in file_names] for file_names in file_matrix
]
env_names = ["Taxi"]  

plot_files(file_matrix, env_names)
