ar_b=(50000 10000 30000)
ar_lr=(0.005 0.01 0.02)

maxb=1
maxlr=3

for (( i=0; i <$maxb; i=i+1)) do
 for (( j=0; j <$maxlr; j=j+1)) do
   python pg_train.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b ${ar_b[$i]} -lr ${ar_lr[$j]} -rtg --nn_baseline --exp_name hc_b${ar_b[$i]}_r${ar_lr[$j]}
 done
done
