
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import numpy as np
import os
import re
import tempfile
import tarfile
from utils.utils import *
import plotly.express as px
import plotly.figure_factory as ff
from gridfs import GridFS
from io import BytesIO
import matplotlib.pyplot as plt
#import seaborn as sns

st.set_page_config(
    page_title="Register",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn model register app!"
    }
)

st.title('SDNN Models Register')

client = init_connection()
db = client.sdnn_model_zoo
## pull models from data base
#models_gridfs = GridFS(db, 'models')
#items = models_collection.find({"name": "yolov6s", "device": "aipu"})
host_ip = "192.168.200.251"
url_pattern = "^http://" + host_ip + ":8092/(?:[0-9a-zA-Z._]+/){1,}"
display_pattern = "http://" + host_ip + ":8092/(?:[0-9a-zA-Z._]+/){1,}(.*?)$"
dataset_display_pattern = "http://" + host_ip + ":8092/(?:[0-9a-zA-Z._]+/){1,}(.*?)/$"

model_category = ["classification_2d", "classification_3d",
        "object_detection_2d", "object_detection_3d",
        "semantic_segmentation", "instance_segmentation", "panoptic_segmentation",
        "others"]

acc_tab, test_tab, client_tab = st.tabs(["Accuracy Models", "Test Models",
    "Client Models"])

with acc_tab:
    st.markdown("***Accuracy*** and ***Performance*** :rainbow[*Model Zoo*]")
    name_map={
        "Name": "name",
        "Category": "category",
        "InputName": "input_name",
        "InputShape": "input_shape",
        "URL": "model_url",
        "CFG": "cfg_url",
        "Dataset": "calib_url",
        "Valid": "valid_url",
    }

    models_collection = db['sdrv']
    items = models_collection.find()
    items = list(items)
    model_df_dict = []
    for item in items:
        model_dict = {
            "Name": item["name"] if "name" in item and item["name"] else "unknow",
            "Category": item["category"] if "category" in item and item["category"] else "unknow",
            "InputName": item["input_name"] if "input_name" in item else "",
            "InputShape": item["input_shape"] if "input_shape" in item else "",
            "URL": item["model_url"] if "model_url" in item else "",
            "CFG": item["cfg_url"] if "cfg_url" in item else "",
            "Dataset": item["calib_url"] if "calib_url" in item else "",
            "Valid": item["valid_url"] if "valid_url" in item else "",
        }
        model_df_dict.append(model_dict)

    if not "df" in ss:
        if len(model_df_dict) == 0:
            ss.df = pd.DataFrame(columns=name_map.keys())
        else:
            ss.df = pd.DataFrame(model_df_dict)

    max_url_chars=300
    config = {
        "Name": st.column_config.TextColumn(
                "Model Name",
                help="The name of model 🎈",
                max_chars=50,
                validate="^\w+$",
                required=True,
            ),
        "Category": st.column_config.SelectboxColumn(
                "Model Category",
                help="The category of the model",
                width="medium",
                options=model_category,
                required=True,
            ),
        "InputName": st.column_config.TextColumn(
                "Input Name",
                help="The name of model input 🎈",
                max_chars=200,
                #validate="^\w+$",
                required=True,
            ),
        "InputShape": st.column_config.TextColumn(
                "Input Shape",
                help="The shape of model input 🎈",
                max_chars=200,
                validate="^\[\d+(?:,\s*\[?\d+\]?)*\]$",
                required=True,
            ),
        "URL": st.column_config.LinkColumn(
                "Model URL",
                help="The URL of model file",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=display_pattern,
                required=True,
            ),
        "CFG": st.column_config.LinkColumn(
                "CFG URL",
                help="The URL of model sdnn cfg file",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=display_pattern
            ),
        "Dataset": st.column_config.LinkColumn(
                "Calib Dataset URL",
                help="The URL of model quant calib dataset",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=dataset_display_pattern
            ),
        "Valid": st.column_config.LinkColumn(
                "Valid Dataset URL",
                help="The URL of model quant valid dataset",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=dataset_display_pattern
            ),
    }

    edit_df = st.data_editor(ss.df, column_config=config, use_container_width=True, key="models", num_rows="dynamic")
    #st.write(ss)
    def model_update():
        add_rows = ss.models["added_rows"]
        if not len(add_rows) == 0:
            for r in add_rows:
                items = models_collection.find({"name": r["Name"]})
                items = list(items)
                if len(items) == 0:
                    model_info_obj = {}
                    model_info_obj["name"]=""
                    model_info_obj["category"]=""
                    model_info_obj["input_name"]=""
                    model_info_obj["input_shape"]=""
                    model_info_obj["model_url"]=""
                    model_info_obj["cfg_url"]=""
                    model_info_obj["calib_url"]=""
                    model_info_obj["valid_url"]=""
                    for k,v in r.items():
                        if k == "_index":
                            continue
                        model_info_obj[name_map[k]] = v
                    db_id = models_collection.insert_one(model_info_obj).inserted_id
                else:
                    replace_list = [{'$set':{name_map[k]:v}} for k,v in r.items()]
                    for rp in replace_list:
                        models_collection.update_many({'name': r["Name"]}, rp)

    st.button("Update Accuracy Model", on_click=model_update)

    del_rows = ss.models["deleted_rows"]
    edt_rows = ss.models["edited_rows"]

    if not len(del_rows) == 0:
        for d in del_rows:
            model_name = ss.df.iloc[d]["Name"]
            models_collection.find_one_and_delete({'name':model_name})

    if not len(edt_rows) == 0:
        for k,v in edt_rows.items():
            model_name = ss.df.iloc[k]["Name"]
            for m,n in v.items():
                models_collection.update_many({'name':model_name}, {'$set':{name_map[m]:n}})


with test_tab:
    st.markdown("***Test*** case :rainbow[*Model Zoo*]")
    test_models_collection = db['test']
    items = test_models_collection.find()
    items = list(items)
    model_df_dict = []
    for item in items:
        model_dict = {
            "name": item["name"] if "name" in item and item["name"] else "unknow",
            "category": item["category"] if "category" in item and item["category"] else "unknow",
            "input_name": item["input_name"] if "input_name" in item else "",
            "input_shape": item["input_shape"] if "input_shape" in item else "",
            "vm": item["vm"] if "vm" in item else "",
            "model_url": item["model_url"] if "model_url" in item else "",
            "cfg_url": item["cfg_url"] if "cfg_url" in item else "",
            "calib_url": item["calib_url"] if "calib_url" in item else "",
            "valid_url": item["valid_url"] if "valid_url" in item else "",
        }
        model_df_dict.append(model_dict)

    if not "test_df" in ss:
        if len(model_df_dict) == 0:
            ss.test_df = pd.DataFrame(columns=["name", "category",
                "input_name", "input_shape", "vm", "model_url", "cfg_url",
                "calib_url", "valid_url"])
        else:
            ss.test_df = pd.DataFrame(model_df_dict)

    max_url_chars=300
    config = {
        "name": st.column_config.TextColumn(
                "Model Name",
                help="The name of model 🎈",
                max_chars=50,
                validate="^\w+$",
                required=True,
            ),
        "category": st.column_config.SelectboxColumn(
                "Model Category",
                help="The category of the model",
                width="medium",
                options=model_category,
                required=True,
            ),
        "input_name": st.column_config.TextColumn(
                "Input Name",
                help="The name of model input 🎈",
                #max_chars=200,
                #validate="^[0-9a-zA-Z_]+$",
                required=True,
            ),
        "input_shape": st.column_config.TextColumn(
                "Input Shape",
                help="The shape of model input 🎈",
                #max_chars=200,
                validate="^\[\d+(?:,\s*\[?\d+\]?)*\]$",
                required=True,
            ),
        "vm": st.column_config.CheckboxColumn(
                "VM",
                help="The mode of model build 🎈",
                default=False,
                required=True,
            ),
        "model_url": st.column_config.LinkColumn(
                "Model URL",
                help="The URL of model file",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=display_pattern,
                required=True,
            ),
        "cfg_url": st.column_config.LinkColumn(
                "CFG URL",
                help="The URL of model sdnn cfg file",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=display_pattern
            ),
        "calib_url": st.column_config.LinkColumn(
                "Calib Dataset URL",
                help="The URL of model quant calib dataset",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=dataset_display_pattern
            ),
        "valid_url": st.column_config.LinkColumn(
                "Valid Dataset URL",
                help="The URL of model quant valid dataset",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=dataset_display_pattern
            ),
    }

    edit_test_df = st.data_editor(ss.test_df, column_config=config,
            use_container_width=True, key="test_models", num_rows="dynamic")
    #st.write(ss)
    def test_model_update():
        add_rows = ss.test_models["added_rows"]
        if not len(add_rows) == 0:
            for r in add_rows:
                items = test_models_collection.find({"name": r["name"]})
                items = list(items)
                if len(items) == 0:
                    model_info_obj = {}
                    model_info_obj["name"]=""
                    model_info_obj["category"]=""
                    model_info_obj["input_name"]=""
                    model_info_obj["input_shape"]=""
                    model_info_obj["vm"]=""
                    model_info_obj["model_url"]=""
                    model_info_obj["cfg_url"]=""
                    model_info_obj["calib_url"]=""
                    model_info_obj["valid_url"]=""
                    for k,v in r.items():
                        if k == "_index":
                            continue
                        model_info_obj[k] = v
                    db_id = test_models_collection.insert_one(model_info_obj).inserted_id
                else:
                    replace_list = [{'$set':{k:v}} for k,v in r.items()]
                    for rp in replace_list:
                        test_models_collection.update_many({'name': r["name"]}, rp)

    st.button("Update Test Model", on_click = test_model_update)

    del_rows = ss.test_models["deleted_rows"]
    edt_rows = ss.test_models["edited_rows"]

    if not len(del_rows) == 0:
        for d in del_rows:
            model_name = ss.test_df.iloc[d]["name"]
            test_models_collection.find_one_and_delete({'name':model_name})

    if not len(edt_rows) == 0:
        for k,v in edt_rows.items():
            model_name = ss.test_df.iloc[k]["name"]
            for m,n in v.items():
                test_models_collection.update_many({'name':model_name}, {'$set':{m:n}})

with client_tab:
    st.markdown("***Client*** case :rainbow[*Model Zoo*]")
    client_models_collection = db['client']
    items = client_models_collection.find()
    items = list(items)
    model_df_dict = []
    for item in items:
        model_dict = {
            "name": item["name"] if "name" in item and item["name"] else "unknow",
            "category": item["category"] if "category" in item and item["category"] else "unknow",
            "input_name": item["input_name"] if "input_name" in item else "",
            "input_shape": item["input_shape"] if "input_shape" in item else "",
            "model_url": item["model_url"] if "model_url" in item else "",
            "cfg_url": item["cfg_url"] if "cfg_url" in item else "",
            "calib_url": item["calib_url"] if "calib_url" in item else "",
            "valid_url": item["valid_url"] if "valid_url" in item else "",
        }
        model_df_dict.append(model_dict)

    if not "client_df" in ss:
        if len(model_df_dict) == 0:
            ss.client_df = pd.DataFrame(columns=["name", "category",
                "input_name", "input_shape", "model_url", "cfg_url",
                "calib_url", "valid_url"])
        else:
            ss.client_df = pd.DataFrame(model_df_dict)

    max_url_chars=300
    config = {
        "name": st.column_config.TextColumn(
                "Model Name",
                help="The name of model 🎈",
                max_chars=50,
                validate="^\w+$",
                required=True,
            ),
        "category": st.column_config.SelectboxColumn(
                "Model Category",
                help="The category of the model",
                width="medium",
                options=model_category,
                required=True,
            ),
        "input_name": st.column_config.TextColumn(
                "Input Name",
                help="The name of model input 🎈",
                max_chars=200,
                validate="^\w+$",
                required=True,
            ),
        "input_shape": st.column_config.TextColumn(
                "Input Shape",
                help="The shape of model input 🎈",
                max_chars=200,
                validate="^\[\d+(?:,\s*\[?\d+\]?)*\]$",
                required=True,
            ),
        "model_url": st.column_config.LinkColumn(
                "Model URL",
                help="The URL of model file",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=display_pattern,
                required=True,
            ),
        "cfg_url": st.column_config.LinkColumn(
                "CFG URL",
                help="The URL of model sdnn cfg file",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=display_pattern
            ),
        "calib_url": st.column_config.LinkColumn(
                "Calib Dataset URL",
                help="The URL of model quant calib dataset",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=dataset_display_pattern
            ),
        "valid_url": st.column_config.LinkColumn(
                "Valid Dataset URL",
                help="The URL of model quant valid dataset",
                validate=url_pattern,
                max_chars=max_url_chars,
                display_text=dataset_display_pattern
            ),
    }

    client_edit_df = st.data_editor(ss.client_df, column_config=config,
            use_container_width=True, key="client_models", num_rows="dynamic")
    #st.write(ss)
    def client_model_update():
        add_rows = ss.client_models["added_rows"]
        if not len(add_rows) == 0:
            for r in add_rows:
                items = client_models_collection.find({"name": r["name"]})
                items = list(items)
                if len(items) == 0:
                    model_info_obj = {}
                    model_info_obj["name"]=""
                    model_info_obj["category"]=""
                    model_info_obj["input_name"]=""
                    model_info_obj["input_shape"]=""
                    model_info_obj["model_url"]=""
                    model_info_obj["cfg_url"]=""
                    model_info_obj["calib_url"]=""
                    model_info_obj["valid_url"]=""
                    for k,v in r.items():
                        if k == "_index":
                            continue
                        model_info_obj[k] = v
                    db_id = client_models_collection.insert_one(model_info_obj).inserted_id
                else:
                    replace_list = [{'$set':{k:v}} for k,v in r.items()]
                    for rp in replace_list:
                        client_models_collection.update_many({'name': r["name"]}, rp)

    st.button("Update Client Model", on_click=client_model_update)

    del_rows = ss.client_models["deleted_rows"]
    edt_rows = ss.client_models["edited_rows"]

    if not len(del_rows) == 0:
        for d in del_rows:
            model_name = ss.client_df.iloc[d]["name"]
            client_models_collection.find_one_and_delete({'name':model_name})

    if not len(edt_rows) == 0:
        for k,v in edt_rows.items():
            model_name = ss.client_df.iloc[k]["name"]
            for m,n in v.items():
                client_models_collection.update_many({'name':model_name}, {'$set':{m:n}})


