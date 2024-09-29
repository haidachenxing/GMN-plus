import os
import subprocess
import config


# step1: generate ida file and function json
for file in os.listdir(config.STEP1_BINARY_DIR):
    if file.endswith(".so"):
        cmd_str = config.IDA_DIR + " -A -S" + config.STEP1_IDA_SCRIPT_PATH + " " + config.STEP1_IDAFILE_DIR  + os.sep + file
        print (cmd_str)

        p = subprocess.Popen(cmd_str, shell=True)
        p.wait()

subprocess.call(["python"], config.ROOT_DIR + os.sep + "2json.py")

# step2: generate semantic embeddings(Bert)

subprocess.call(["python", config.ROOT_DIR + os.sep + "gen_FIT_sent_bert_vec_means.py"])

# step3: generate function json file and function pair list

subprocess.call(["python", config.ROOT_DIR + os.sep + "generate_json.py"])
subprocess.call(["python", config.ROOT_DIR + os.sep + "generate_all.py"])

# step4: GMN model generate vulnerability analysis results

subprocess.call(["python"], config.ROOT_DIR + os.sep + "train.py")
