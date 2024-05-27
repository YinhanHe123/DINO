#!/bin/bash

# Define session name
SESSION_NAME="DINO_experiments"
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $SESSION_NAME already exists. Killing it."
  tmux kill-session -t $SESSION_NAME
fi

# Start a new tmux session
tmux new-session -d -s $SESSION_NAME

# Arrays of datasets and k values
datasets=('p2p-Gnutella08' 'Wiki-Vote' 'WikiTalk' 'soc-Epinions1' 'Email-EuAll' 'hiv_transmission')
ks=(100 300 500)

# Pane counter to keep track of which pane we're working on
pane_counter=0

# Loop over each dataset
for data in "${datasets[@]}"; do
    # Loop over each k value
    for y in "${ks[@]}"; do
        if [ $pane_counter -ne 0 ]; then
            # Split window vertically for additional commands
            tmux split-window -v -t $SESSION_NAME
            tmux select-layout -t $SESSION_NAME tiled > /dev/null
        fi

        # Form the command with the current dataset and k value
        cmd="python main.py -k $y -d $data"

        # Send the command to the pane
        tmux send-keys -t $pane_counter "$cmd" C-m

        # Increment pane counter
        pane_counter=$((pane_counter + 1))
    done
done

# Ensure all panes are evenly distributed
tmux select-layout -t $SESSION_NAME tiled

# Attach to the session
tmux attach -t $SESSION_NAME