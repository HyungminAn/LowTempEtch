#!/bin/bash

LOGFILE="delete.log"

FILE_PATTERNS=("vasprun.xml" "vaspout.h5" "PROCAR" "DOSCAR" "POTCAR" "OUTCAR_0*" "std_0*" "XDATCAR_0*")

find . -type f \( -false $(for pattern in "${FILE_PATTERNS[@]}"; do echo -o -name "$pattern"; done) \) -print0 |
while IFS= read -r -d '' file; do
    if rm "$file"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Deleted: $file" >> "$LOGFILE"
    fi
done
