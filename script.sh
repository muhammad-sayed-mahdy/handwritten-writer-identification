#!/bin/bash
input="../ascii/forms.txt"
while IFS=' ' read -r f1 f2 f3 f4 f5 f6 f7 f8
do
   mkdir -p ../data/$f2
   mv $f1.png ../data/$f2
   echo $f1
done < "$input"
