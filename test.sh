cd /public/home/jiayanhao/BLIP/

nohup python itm_predict.py --start 39950 --end 39970 --save_result 'result/result_39950-39970.npy'\
     --device 'cuda:0' \
     > log/39950-39970.log 2>&1 &

nohup python itm_predict.py --start 39970 --end 39990 --save_result 'result/result_39970-39990.npy'\
      --device 'cuda:1' \
     > log/39970-39990.log 2>&1 &

nohup python itm_predict.py --start 39990 --end 40010 --save_result 'result/result_39990-40010.npy'\
      --device 'cuda:2' \
     > log/39990-40010.log 2>&1 &

nohup python itm_predict.py --start 40010 --end 40030 --save_result 'result/result_40010-40030.npy'\
      --device 'cuda:3' \
     > log/40010-40030.log 2>&1 &

nohup python itm_predict.py --start 40030 --end 40050 --save_result 'result/result_40030-40050.npy'\
      --device 'cuda:4' \
     > log/40030-40050.log 2>&1 &

nohup python itm_predict.py --start 40050 --end 40070 --save_result 'result/result_40050-40070.npy'\
      --device 'cuda:5' \
     > log/40050-40070.log 2>&1 &

nohup python itm_predict.py --start 40070 --end 40090 --save_result 'result/result_40070-40090.npy'\
      --device 'cuda:6' \
     > log/40070-40090.log 2>&1 &

nohup python itm_predict.py --start 40090 --end 40110 --save_result 'result/result_40090-40110.npy'\
      --device 'cuda:7' \
     > log/40090-40110.log 2>&1 &
