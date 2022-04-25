#bin/bash 
#打印GPU信息 
#执行math.py
python3 ./code/data_preprocess.py
python3 ./code/tokenizer.py
python3 ./code/pre_train.py
python3 ./code/xlnet_model.py
python3 ./code/predict.py
zip -j result.zip /prediction_result/predictions.csv