#!/bin/bash

hosts=`cat $1`
rank=0
path=
key=
nodes=$2
output=$3
master=

for host in $hosts; do
    echo $host
    echo $rank
    ssh -o "StrictHostKeyChecking no" -i $key ubuntu@$host "
	cd $path
	tmux new -d -s pisa_512
	tmux send-keys -t pisa_512:0 './launch_mlperf_512.sh $rank $nodes $output $master' Enter
    " &
    rank=$((rank+1))
done
