#bin/bash 
#打印GPU信息 
#执行math.py
python3 ./code/final_data_process.py
python3 ./code/bert_epoch.py
python3 ./code/predict.py
zip -j result.zip /prediction_result/predictions.csv