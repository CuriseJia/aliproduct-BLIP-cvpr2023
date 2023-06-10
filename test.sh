cd /public/home/jiayanhao/BLIP/

nohup python demo_predict.py --start 39950 --end 39970 --save_result 'result/result_39950-39970.npy'\
     --device 'cuda:0' \
     > log/39950-39970.log 2>&1 &

nohup python demo_predict.py --start 39970 --end 39990 --save_result 'result/result_39970-39990.npy'\
      --device 'cuda:1' \
     > log/39970-39990.log 2>&1 &

nohup python demo_predict.py --start 39990 --end 40010 --save_result 'result/result_39990-40010.npy'\
      --device 'cuda:2' \
     > log/39990-40010.log 2>&1 &

nohup python demo_predict.py --start 40010 --end 40030 --save_result 'result/result_40010-40030.npy'\
      --device 'cuda:3' \
     > log/40010-40030.log 2>&1 &

nohup python demo_predict.py --start 40030 --end 40050 --save_result 'result/result_40030-40050.npy'\
      --device 'cuda:4' \
     > log/40030-40050.log 2>&1 &

nohup python demo_predict.py --start 40050 --end 40070 --save_result 'result/result_40050-40070.npy'\
      --device 'cuda:5' \
     > log/40050-40070.log 2>&1 &

nohup python demo_predict.py --start 40070 --end 40090 --save_result 'result/result_40070-40090.npy'\
      --device 'cuda:6' \
     > log/40070-40090.log 2>&1 &

nohup python demo_predict.py --start 40090 --end 40110 --save_result 'result/result_40090-40110.npy'\
      --device 'cuda:7' \
     > log/40090-40110.log 2>&1 &



# nohup python demo_predict.py --start 2500 --end 2520 --save_result 'result/result_2500-2520.npy'\
#      --device 'cuda:0' \
#      > log/2500-2520.log 2>&1 &

# nohup python demo_predict.py --start 2520 --end 2540 --save_result 'result/result_2520-2540.npy'\
#       --device 'cuda:1' \
#      > log/2520-2540.log 2>&1 &

# nohup python demo_predict.py --start 2540 --end 2560 --save_result 'result/result_2540-2560.npy'\
#       --device 'cuda:2' \
#      > log/2540-2560.log 2>&1 &

# nohup python demo_predict.py --start 2560 --end 2580 --save_result 'result/result_2560-2580.npy'\
#       --device 'cuda:3' \
#      > log/2560-2580.log 2>&1 &

# nohup python demo_predict.py --start 2580 --end 2600 --save_result 'result/result_2580-2600.npy'\
#       --device 'cuda:4' \
#      > log/2580-2600.log 2>&1 &

# nohup python demo_predict.py --start 2600 --end 2620 --save_result 'result/result_2600-2620.npy'\
#       --device 'cuda:5' \
#      > log/2600-2620.log 2>&1 &

# nohup python demo_predict.py --start 2620 --end 2640 --save_result 'result/result_2620-2640.npy'\
#       --device 'cuda:6' \
#      > log/2620-2640.log 2>&1 &

# nohup python demo_predict.py --start 2640 --end 2660 --save_result 'result/result_2640-2660.npy'\
#       --device 'cuda:7' \
#      > log/2640-2660.log 2>&1 &