#!/bin/bash

for f in phy/*; do
    dd of="$f" oflag=nocache conv=notrunc,fdatasync count=0 status=none 
done

for f in rba/*; do
    dd of="$f" oflag=nocache conv=notrunc,fdatasync count=0 status=none 
done
