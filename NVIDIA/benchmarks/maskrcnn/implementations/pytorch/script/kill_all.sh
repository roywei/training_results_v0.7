#!/bin/bash

hosts=`cat $1`
key=

for host in $hosts; do
    echo "$host"
    ssh -i $key ubuntu@$host "
      killall -9 python
      tmux kill-server
      docker stop mlperf_training
    " &
done
