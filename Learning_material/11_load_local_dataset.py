'''

数据格式 	类型参数 	加载的指令
CSV & TSV 	csv 	load_dataset("csv", data_files="my_file.csv")
Text files 	text 	load_dataset("text", data_files="my_file.txt")
JSON & JSON Lines 	json 	load_dataset("json", data_files="my_file.jsonl")
Pickled DataFrames 	pandas 	load_dataset("pandas", data_files="my_dataframe.pkl")


# actual data dict json format

DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})

'''


from datasets import load_dataset
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")


# directly use gz 
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}

# auto load the train and test dataset
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset



# load remote dataset

url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
