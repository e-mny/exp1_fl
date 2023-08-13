#!/bin/bash
num_clients=2

# Federated Training
echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

sequence=$(seq 1 $num_clients)

for i in $sequence; do
    echo "Starting client $i"
    python client.py &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

# Centralized Training
python cifar.py
