import re
import matplotlib.pyplot as plt

# Load log file
log_file_path = "training_logs_part2_bots.txt"

def get_metrics(log_file_path):
    
    # lst to store parsed data
    episodes = []
    global_steps = []
    progresses = []
    deviations = []
    lap_times = []

    pattern = re.compile(
    r"Episode (\d+)/\d+ - Global Steps (\d+) - Return: [\d.]+ - Progress: ([\d.]+)% - Deviation: ([\d.]+) - Length: \d+ - Lap Time: ([\d.]+|nan) sec")




    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                episode = int(match.group(1))
                step = int(match.group(2))
                progress = float(match.group(3))
                deviation = float(match.group(4))
                lap_time = float(match.group(5)) if match.group(4) != 'nan' else None

                episodes.append(episode)
                global_steps.append(step)
                progresses.append(progress)
                deviations.append(deviation)
                lap_times.append(lap_time)
    return (episodes,global_steps,progresses,deviations,lap_times)







#####

def generate_plots(race_type,window_size,episodes,global_steps,progresses,deviations,lap_times):

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    num_episodes=len(episodes)

    
    window_size = window_size  
    df = pd.DataFrame({'global_steps': global_steps, 'progress': progresses,'deviation': deviations,'lap_time': lap_times})

    #progress plot

    smoothed_progress = df['progress'].rolling(window=window_size, center=True).mean()

    
    valid_steps_progress = df['global_steps'][smoothed_progress.notna()]
    smoothed_progress = smoothed_progress.dropna()

    # plot
    plt.figure(figsize=(14, 6))  
    plt.plot(valid_steps_progress, smoothed_progress, label='Smoothed Progress', color='blue')
    plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label="Threshold (100% Progress)")

    plt.xlabel('Global Steps('+str(num_episodes)+' Episodes)')
    plt.ylabel('Progress (%)')
    if race_type=="htb":
        head_t="Head-to-Bot:"
    elif race_type=="tt":
        head_t="Time-Trial:"
    elif race_type=='obs':
        head_t="Object-Avoidance:"
    
    plt.title(head_t+' Smoothed Progress vs Global Steps')

    plt.xticks(rotation=45)
    plt.locator_params(axis='x', nbins=30)


    plt.xlim(left=valid_steps_progress.min())  

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(race_type+"_smoothed_progress_vs_globalsteps.png")
    plt.clf()

    #deviation plot


    smoothed_deviation = df['deviation'].rolling(window=window_size, center=True).mean()


    valid_steps_deviation = df['global_steps'][smoothed_deviation.notna()]
    smoothed_deviation = smoothed_deviation.dropna()

 
    plt.figure(figsize=(14, 6))  
    plt.plot(valid_steps_deviation, smoothed_deviation, label='Smoothed Deviation', color='red')

    plt.xlabel('Global Steps('+str(num_episodes)+' Episodes)')
    plt.ylabel('Deviation')
    if race_type=="htb":
        head_t="Head-to-Bot:"
    elif race_type=="tt":
        head_t="Time-Trial:"
    elif race_type=='obs':
        head_t="Object-Avoidance:"
    
    plt.title(head_t+' Smoothed Center line Deviation vs Global Steps')

    plt.xticks(rotation=45)
    plt.locator_params(axis='x', nbins=30)

    

    plt.xlim(left=valid_steps_deviation.min())  

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(race_type+"_smoothed_deviation_vs_globalsteps.png")
    plt.clf()

    #lap times plot


    smoothed_lap_time = df['lap_time'].rolling(window=2, center=True).mean()

    valid_steps_lap_time = df['global_steps'][smoothed_lap_time.notna()]
    smoothed_lap_time = smoothed_lap_time.dropna()

    if len(smoothed_lap_time)>0:

      
        plt.figure(figsize=(14, 6))  
        plt.plot(valid_steps_lap_time, smoothed_lap_time, label='Lap-times', color='green')

        plt.xlabel('Global Steps('+str(num_episodes)+' Episodes)')
        plt.ylabel('Lap times (sec)')
        if race_type=="htb":
            head_t="Head-to-Bot:"
        elif race_type=="tt":
            head_t="Time-Trial:"
        elif race_type=='obs':
            head_t="Object-Avoidance:"
        
        plt.title(head_t+'Smoothed Lap-times vs Global Steps')

        plt.xticks(rotation=45)
        plt.locator_params(axis='x', nbins=30)

      
        plt.xlim(left=valid_steps_lap_time.min())  

        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(race_type+"_laptime_vs_globalsteps.png")
        plt.clf()


log_file_path = "training_logs_part2_bots.txt"
episodes,global_steps,progresses,deviations,lap_times=get_metrics(log_file_path)

generate_plots(race_type='htb',window_size=10,episodes=episodes,global_steps=global_steps,progresses=progresses,deviations=deviations,lap_times=lap_times)

log_file_path = "training_logs_part1_tt.txt"
episodes,global_steps,progresses,deviations,lap_times=get_metrics(log_file_path)

generate_plots(race_type='tt',window_size=10,episodes=episodes,global_steps=global_steps,progresses=progresses,deviations=deviations,lap_times=lap_times)


log_file_path = "training_logs_part2_obs copy 3.txt"
episodes,global_steps,progresses,deviations,lap_times=get_metrics(log_file_path)

generate_plots(race_type='obs',window_size=10,episodes=episodes,global_steps=global_steps,progresses=progresses,deviations=deviations,lap_times=lap_times)


##### plot eval metrics

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load json
with open('/home/ubuntu/P4_deepracer/evaluations/head_to_bot-ppo_agent.json', "r") as f:
    data = json.load(f)


rows = []
for track, values in data.items():
    for prog, lap in zip(values["progress"], values["lap_time"]):
        lap_time = float(lap) if str(lap).lower() != "nan" else np.nan
        rows.append({
            "Track": track,
            "Progress": prog,
            "Lap Time": lap_time
        })

df = pd.DataFrame(rows)


fig, axs = plt.subplots(1, 2, figsize=(14, 6))

custom_palette = ["#3b6ba5", "#cd804a", "#6b8e58"]  # brown, brown, green

# progress boxplot
sns.boxplot(
    data=df, x="Track", y="Progress", ax=axs[0],
    palette=custom_palette,
    showmeans=True,
    meanprops={
        "marker": "^",  
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markeredgewidth": 2
    },
    flierprops={'marker': 'x'}
)
axs[0].set_title("Progress%")
axs[0].tick_params(axis='x', rotation=45)

# laptime box
sns.boxplot(
    data=df, x="Track", y="Lap Time", ax=axs[1],
    palette=custom_palette,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markeredgewidth": 2
    },
    flierprops={'marker': 'x'}
)

axs[1].set_title("Lap Time")
axs[1].set_ylabel("Lap Time (seconds)")
axs[1].tick_params(axis='x', rotation=45)


plt.suptitle("Head-to-Bot Evaluation Plots: Progress & Lap Time by Track", fontsize=14)
plt.tight_layout()
plt.savefig("htb_evaluation_boxplots.png")


### TIME-TRIAL


import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


with open('/home/ubuntu/P4_deepracer/evaluations/time_trial-ppo_agent.json', "r") as f:
    data = json.load(f)


rows = []
for track, values in data.items():
    for prog, lap in zip(values["progress"], values["lap_time"]):
        lap_time = float(lap) if str(lap).lower() != "nan" else np.nan
        rows.append({
            "Track": track,
            "Progress": prog,
            "Lap Time": lap_time
        })

df = pd.DataFrame(rows)


fig, axs = plt.subplots(1, 2, figsize=(14, 6))

custom_palette = ["#3b6ba5", "#cd804a", "#6b8e58"]  # brown, brown, green

sns.boxplot(
    data=df, x="Track", y="Progress", ax=axs[0],
    palette=custom_palette,
    showmeans=True,
    meanprops={
        "marker": "^",  # triangle marker
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markeredgewidth": 2
    },
    flierprops={'marker': 'x'}
)
axs[0].set_title("Progress%")
axs[0].tick_params(axis='x', rotation=45)

sns.boxplot(
    data=df, x="Track", y="Lap Time", ax=axs[1],
    palette=custom_palette,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markeredgewidth": 2
    },
    flierprops={'marker': 'x'}
)
axs[1].set_title("Lap Time")
axs[1].set_ylabel("Lap Time (seconds)")
axs[1].tick_params(axis='x', rotation=45)

plt.suptitle("Time-trial Evaluation Plots: Progress & Lap Time by Track", fontsize=14)
plt.tight_layout()
plt.savefig("tt_evaluation_boxplots.png")

##OBS-AVOIDANCE


import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open('/home/ubuntu/P4_deepracer/evaluations/obstacle_avoidance-ppo_agent.json', "r") as f:
    data = json.load(f)

rows = []
for track, values in data.items():
    for prog, lap in zip(values["progress"], values["lap_time"]):
        lap_time = float(lap) if str(lap).lower() != "nan" else np.nan
        rows.append({
            "Track": track,
            "Progress": prog,
            "Lap Time": lap_time
        })

df = pd.DataFrame(rows)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

custom_palette = ["#3b6ba5", "#cd804a", "#6b8e58"]  # brown, brown, green

sns.boxplot(
    data=df, x="Track", y="Progress", ax=axs[0],
    palette=custom_palette,
    showmeans=True,
    meanprops={
        "marker": "^",  # triangle marker
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markeredgewidth": 2
    },
    flierprops={'marker': 'x'}
)
axs[0].set_title("Progress%")
axs[0].tick_params(axis='x', rotation=45)

sns.boxplot(
    data=df, x="Track", y="Lap Time", ax=axs[1],
    palette=custom_palette,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markeredgewidth": 2
    },
    flierprops={'marker': 'x'}
)
axs[1].set_title("Lap Time")
axs[1].set_ylabel("Lap Time (seconds)")
axs[1].tick_params(axis='x', rotation=45)

plt.suptitle("Obstacle-Avoidance Evaluation Plots: Progress & Lap Time by Track", fontsize=14)
plt.tight_layout()
plt.savefig("obs_evaluation_boxplots.png")