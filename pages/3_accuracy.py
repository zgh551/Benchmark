
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from utils.utils import filter_dataframe
from utils.utils import init_connection, get_next_items, get_models
#from gridfs import GridFS

st.set_page_config(
    page_title="Accuracy",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn accuracy benchmark summarize app!"
    }
)

st.title('SDNN Accuracy Benchmark Summarize')

client = init_connection()
db = client.sdnn_accuracy
items = db.classification.find()
items = list(items)

topk_dict = {}

for item in items:
    topk_dict[item["name"]][item["device"]]["top1"] = item["top1"] * 100
    topk_dict[item["name"]][item["device"]]["top5"] = item["top5"] * 100

for item in topk_dict:
    col1, col2, col3, col4 = st.columns(4)
    col1.header(item["name"])
    col2.header(item["device"])
    col3.metric("TOP1", "{:.2f}%".format(item["top1"] * 100), "-8%")
    col4.metric("TOP5", "{:.2f}%".format(item["top5"] * 100), "4%")

    #with st.expander("TOPK json"):
    #    st.write(item["topk_json"])


