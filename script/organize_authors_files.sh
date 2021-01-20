#!/bin/bash
input="forms.txt"
mkdir -p ../data_tune
while IFS=' ' read -r f1 f2 f3 f4 f5 f6 f7 f8
do
   mkdir -p ../data_tune/$f2
   mv $f1.png ../data_tune/$f2
   echo $f1
done < "$input"
