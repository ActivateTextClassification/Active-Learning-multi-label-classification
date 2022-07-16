import pandas as pd

path1 = "dataset/wos5736/WOS11967/WOS11967_all.csv"
path2 = "dataset/wos5736/WOS11967/WOS11967_train.csv"
path3 = "dataset/wos5736/WOS11967/WOS11967_test.csv"
path4 = "dataset/wos5736/WOS11967/WOS11967_unlabeled.csv"


arxiv_tra1 = pd.read_csv(path1)
arxiv_tra2 = pd.read_csv(path2)
arxiv_tra3 = pd.read_csv(path3)
arxiv_tra4 = pd.read_csv(path4)

print(arxiv_tra1.head())

arxiv_tra1 = arxiv_tra1.rename(columns={"label": "labels"})
arxiv_tra2 = arxiv_tra2.rename(columns={"label": "labels"})
arxiv_tra3 = arxiv_tra3.rename(columns={"label": "labels"})
arxiv_tra4 = arxiv_tra4.rename(columns={"label": "labels"})


arxiv_tra1.to_csv(path1, index=False)
arxiv_tra2.to_csv(path2, index=False)
arxiv_tra3.to_csv(path3, index=False)
arxiv_tra4.to_csv(path4, index=False)