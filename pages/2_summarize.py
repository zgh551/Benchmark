
import streamlit as st
import pandas as pd
import numpy as np
from utils.utils import filter_dataframe
from utils.utils import init_connection, get_next_items, get_models

st.set_page_config(
    page_title="Benchmark Summarize",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn benchmark summarize app!"
    }
)

st.title('SDNN Benchmark Summarize')

client = init_connection()
items = get_models(client)

df_dict = {}
df_dict["name"] = []
df_dict["input"] = []
df_dict["output"] = []
df_dict["chip"] = []
df_dict["sdk"] = []
df_dict["os"] = []
df_dict["ptg"] = []
df_dict["device"] = []
df_dict["quant"] = []
df_dict["npu"] = []
df_dict["dequant"] = []
df_dict["date"] = []
#df_dict["number"] = []

def get_shape(data: list):
    shape_str = ""
    for d in data:
        shape_str += "(%s)%s " % (d['dtype'], str(d['shape']))
    return shape_str

def get_mean(data: list):
    np_data = np.array(data)
    sorted_data = np.sort(np_data)
    median_index = np.median(sorted_data)
    if len(sorted_data) % 2 == 1:
        return sorted_data[int(median_index)]
    else:
        return (sorted_data[int(median_index)] + sorted_data[int(median_index) -1 ]) * 0.5


for item in items:
    df_dict["name"].append(item['name'])
    input_str = ""
    df_dict["input"].append(get_shape(item['deploy_json'][0]['inputs']))
    df_dict["output"].append(get_shape(item['deploy_json'][0]['outputs']))
    df_dict["chip"].append(item['chip'])
    df_dict["device"].append(item['device'])
    df_dict["sdk"].append(item['sdk'])
    df_dict["os"].append(item['os'])
    df_dict["ptg"].append(item['ptg'])
    df_dict["date"].append(item['date'])
    data_dict = {"npu":None, "quant":None, "dequant":None}
    for k, v in item["data"].items():
        if "compass" in k:
            data_dict["npu"] = get_mean(v)
        elif "divide" in k:
            data_dict["quant"] = get_mean(v)
        elif "multiply" in k:
            data_dict["dequant"] = get_mean(v)
        elif "cast" in k:
            data_dict["dequant"] = get_mean(v)
        else:
            data_dict["dequant"] = get_mean(v)
    for k,v in data_dict.items():
        df_dict[k].append(v)


df = pd.DataFrame(df_dict)

with st.sidebar:
    filter_df = filter_dataframe(df)

with st.container(border=True):
    st.dataframe(filter_df,
        column_config={
            "name": "Model Name",
            "input": "Input Shape",
            "output": "Onput Shape",
            "chip": "Chip",
            "device": "Device",
            "sdk": "SDK",
            "os": "OS",
            "ptg": "PTG",
            "npu": "NPU",
            "quant": "Quant",
            "dequant": "Dequant",
            "date": "Date",
        },
        height = 700,
        use_container_width = True,)
