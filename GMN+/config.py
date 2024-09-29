import os

IDA_DIR = "/home/ubuntu/Desktop/ida/IDA_P/idaq64"

SOFTWARE_NAME = "curl_v2"
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
VUL_JSONSET_PATH = ROOT_DIR + os.sep + "vul_json"

# step1
STEP1_BINARY_DIR = ROOT_DIR + os.sep + "binary_file"
STEP1_IDAFILE_DIR = ROOT_DIR + os.sep + "ida_file"
STEP1_IDA_SCRIPT_PATH = ROOT_DIR + os.sep + "3LACFG_Generator" + os.sep + "my_preprocess.py"
STEP1_JSON_DIR = ROOT_DIR + os.sep + "json" + os.sep + SOFTWARE_NAME
# step2

STEP2_SOFTWARE_NAME = SOFTWARE_NAME
STEP2_TOKENIER_DIR = "/home/ubuntu/Desktop/Bert/add_token/new_vo/xml_2.9.2/model"
STEP2_INSTRUCTIONS_DIR = ROOT_DIR + os.sep + "json"
STEP2_SAVE_DIR = ROOT_DIR + os.sep + "embedding"
# STEP2_MODEL_DIR = "/home/ubuntu/Desktop/Bert/finetune_bert/data/model_save/changebert/glpk/new_data3.csv10_23_16_38"
STEP2_MODEL_DIR = "/home/ubuntu/Desktop/Bert/finetune_bert/data/model_save/changebert/xml_2.9.2/new_data3.csv11_12_17_09"
# step3
STEP3_VUL_EMBEDDING_DIR = STEP2_SAVE_DIR
STEP3_DATASET_EMBEDDING_DIR = ROOT_DIR + os.sep + "embedding" + os.sep + "dataset"
STEP3_JSON_DIR = ROOT_DIR + os.sep + "json" + os.sep + STEP2_SOFTWARE_NAME
STEP3_SAVE_DIR = ROOT_DIR + os.sep + "function_rep" + os.sep + STEP2_SOFTWARE_NAME
STEP3_FUNC_LIST_DIR = ROOT_DIR + os.sep + "function_list" + os.sep + STEP2_SOFTWARE_NAME
STEP3_DATASET_DIR = ROOT_DIR + os.sep + "vul_json"

# step4
STEP4_DATESET_JSON = STEP3_SAVE_DIR + os.sep + "func_date.json"
STEP4_FUNCPAIR_LIST = STEP3_FUNC_LIST_DIR
STEP4_MODEL_DIR = ROOT_DIR + os.sep + "model" + os.sep + "model_glpk.pth"
STEP4_RESULT_DIR = ROOT_DIR + os.sep + "detection_result"
