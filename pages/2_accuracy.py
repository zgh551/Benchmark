
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from utils.utils import *

st.set_page_config(
    page_title="Accuracy",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn accuracy benchmark summarize app!"
    }
)

client = init_connection()

st.title('SDNN Accuracy Benchmark Summarize')

cls_tab, od_tab = st.tabs(["Classification 2D", "Object Detection 2D"])

with cls_tab:
    db = client.sdnn_benchmark
    items = db.classification.find()
    items = list(items)

    classification_df_dict = {}
    classification_df_dict["name"] = []
    classification_df_dict["device"] = []
    classification_df_dict["sdk"] = []
    classification_df_dict["top1"] = []
    classification_df_dict["top5"] = []
    classification_df_dict["date"] = []

    for item in items:
        classification_df_dict["date"].append(item["date"])
        classification_df_dict["name"].append(item["name"])
        classification_df_dict["device"].append(item["device"])
        classification_df_dict["sdk"].append(item["sdk"])
        classification_df_dict["top1"].append(item["top1"])
        classification_df_dict["top5"].append(item["top5"])

    class_df = pd.DataFrame(classification_df_dict)

    for name in class_df["name"].unique():
        df_name = class_df[class_df["name"].str.contains(name)]
        #st.write(df_name)
        with st.container(border=True):
            d_num = len(df_name)
            name_col1, data_col2 = st.columns((1, 3))
            name_col1.container(height=35*d_num, border=False)
            name_col1.title(name)
            with data_col2:
                for index, row in df_name.iterrows():
                    #print(dir(d))
                    with st.container(border=True):
                        sdk_col, device_col, top1_col, top5_col = st.columns((1, 1, 1, 1))
                        sdk_col.header(row['sdk'])
                        device_col.header(row['device'])
                        if row['device'] == 'cpu':
                            top1_delta = 0
                            top5_delta = 0
                        else:
                            df_cpu = df_name[df_name["device"].str.contains('cpu')]
                            if len(df_cpu) > 0:
                                top1_delta = row['top1'] - df_cpu.iloc[0]['top1']
                                top5_delta = row['top5'] - df_cpu.iloc[0]['top5']

                        top1_col.metric("TOP1",
                                    value = "{:.2f}%".format(row['top1'] * 100),
                                    delta = "{:.2f}%".format(top1_delta * 100),
                                    delta_color = "off" if top1_delta == 0 else "normal")
                        top5_col.metric("TOP5",
                                    value = "{:.2f}%".format(row["top5"] * 100),
                                    delta = "{:.2f}%".format(top5_delta * 100),
                                    delta_color = "off" if top1_delta == 0 else "normal")

    #with st.expander("TOPK json"):
    #    st.write(item["topk_json"])
    with st.expander("TOPK Table"):
        with st.container(border=True):
            st.dataframe(class_df,
                column_config={
                    "name": "Model Name",
                    "device": "Device",
                    "sdk": "SDK",
                    "top1": "Top1",
                    "top5": "Top5",
                    "date": "Date Time",
                },
                use_container_width = True,)

with od_tab:
    db = client.sdnn_benchmark
    items = db.detecttion_2d.find()
    items = list(items)

    detection_2d_df_dict = {}
    detection_2d_df_dict["name"] = []
    detection_2d_df_dict["sdk"] = []
    detection_2d_df_dict["device"] = []
    detection_2d_df_dict["quant_bits"] = []
    detection_2d_df_dict["number"] = []
    detection_2d_df_dict["AP(0.5:0.95)"] = []
    detection_2d_df_dict["AP(0.5)"] = []
    detection_2d_df_dict["AP(0.75)"] = []
    detection_2d_df_dict["date"] = []

    for item in items:
        detection_2d_df_dict["date"].append(item["date"])
        detection_2d_df_dict["name"].append(item["name"])
        detection_2d_df_dict["sdk"].append(item["sdk"])
        detection_2d_df_dict["quant_bits"].append(item["quant_bits"])
        detection_2d_df_dict["device"].append(item["device"])
        detection_2d_df_dict["number"].append(item["number"])
        detection_2d_df_dict["AP(0.5:0.95)"].append(item["mAP"][0])
        detection_2d_df_dict["AP(0.5)"].append(item["mAP"][1])
        detection_2d_df_dict["AP(0.75)"].append(item["mAP"][2])
        #print(detection_2d_df_dict)


    od_df = pd.DataFrame(detection_2d_df_dict)
    #st.dataframe(od_df)

    st.dataframe(od_df,
        column_config={
            "name": "Model Name",
            "sdk": "SDK",
            "quant_bits": "QBit",
            "device": "Device",
            "number": "Number",
            "AP(0.5:0.95)": "AP(0.5:0.95)",
            "AP(0.5)": "AP(0.5)",
            "AP(0.75)": "AP(0.75)",
            "date": "Date Time",
        },
        use_container_width = True,)

