import os
import json
import glob


read_inst_path = "/home/ubuntu/Desktop/2tdisk/zth/ABERT/Bert/add_token/corpus/json"
save_dir = "/home/ubuntu/Desktop/2tdisk/zth/ABERT/Bert/add_token/corpus/std_inst_files"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# for json_file in os.listdir(read_inst_path):
#     arch_path = read_inst_path + os.sep + json_file
arch_paths = glob.glob(read_inst_path+ os.sep + '*')
for arch_path in arch_paths:
    json_name = arch_path.split(os.sep)[-1][:-5]
    save_file = save_dir + os.sep + json_name + '.txt'
    with open(save_file,'a') as fp_save:
        with open(arch_path,'r')as fp_load_json:
            for line in fp_load_json.readlines():
                infos = json.loads(line)
                for info in infos['features']:  #info : feature of a block
                    sentence = ' '.join(info[12]) + '\n'  #get standardized instruction of a block
                    fp_save.write(sentence)