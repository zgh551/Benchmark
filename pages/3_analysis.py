
import streamlit as st
from streamlit import session_state as ss
from bson.objectid import ObjectId
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
    page_title="Analysis",
    page_icon="👋",
    layout="wide",
    menu_items={
        'About': "# sdnn model accuracy analysis app!"
    }
)

st.title('SDNN Compiled Model Accuracy Analysis')

client = init_connection()
items = get_compiled_models(client)

df_dict = {}
df_dict["show"] = []
df_dict["name"] = []
df_dict["category"] = []
df_dict["sdk"] = []
df_dict["device"] = []
df_dict["os"] = []
df_dict["acc_level"] = []
df_dict["bit_quant"] = []
## other info no display
df_dict["date"] = []
df_dict["model_tar_id"] = []
df_dict["opt_dump_id"] = []
df_dict["opt_cfg_id"] = []
df_dict["opt_log_id"] = []
df_dict["opt_json_id"] = []
df_dict["opt_txt_id"] = []

for item in items:
    df_dict["show"].append(False)
    df_dict["name"].append(item['name'])
    df_dict["category"].append(item['category'])

    df_dict["sdk"].append(item["sdk"])
    df_dict["device"].append(item["device"])
    df_dict["os"].append(item["os"])
    df_dict["acc_level"].append(item["acc_level"])
    df_dict["bit_quant"].append(item["quant_bits"])
    ## other info no display
    df_dict["date"].append(item["date"])
    df_dict["model_tar_id"].append(item["model_tar"].binary)
    df_dict["opt_dump_id"].append(item["opt_dump"].binary)
    df_dict["opt_cfg_id"].append(item["opt_cfg"].binary)
    df_dict["opt_log_id"].append(item["opt_log"].binary)
    df_dict["opt_json_id"].append(item["opt_json"].binary)
    df_dict["opt_txt_id"].append(item["opt_txt"].binary)

df = pd.DataFrame(df_dict)

with st.sidebar:
    filter_df = filter_dataframe(df)

    st.header("Option Info", divider='rainbow')
    opt_json_show = st.toggle('Show OPT json?')
    opt_log_show = st.toggle('Show OPT Log?')
    opt_cfg_show = st.toggle('Show OPT CFG?')
    opt_txt_show = st.toggle('Show OPT IR?')

with st.container(border=True):
    st.data_editor(filter_df,
        column_order=["show", "name", "category", "input_name", "input_shape",
            "sdk", "device", "os", "acc_level", "bit_quant", "date"],
        disabled=["name", "category", "input_name", "input_shape",
            "sdk", "device", "os", "acc_level", "bit_quant", "date"],
        column_config={
            "show": "Show",
            "name": "Model Name",
            "input_name": "Input Name",
            "input_shape": "Input Shape",
            "sdk": "SDK",
            "device": "Device",
            "os": "OS",
            "acc_level": "Level",
            "bit_quant": "Quant Bit",
            "date": "Date",
        },
        hide_index = True,
        key="accuracy_df",
        use_container_width = True,)

edt_rows = ss.accuracy_df["edited_rows"]

is_show_df_index_list = []
if not len(edt_rows) == 0:
    for k,v in edt_rows.items():
        is_show = bool(v["show"])
        if is_show:
            is_show_df_index_list.append(k)

if len(is_show_df_index_list) > 0:
    ## record selected performance info
    model_accuracy_info_dict = {}
    for index in is_show_df_index_list:
        ## multi data compare add some info
        aux_lable_name = ""
        if len(filter_df["sdk"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["sdk"]
        if len(filter_df["acc_level"].unique()) >= 2:
            aux_lable_name += "-" + str(filter_df.iloc[index]["acc_level"])
        if len(filter_df["bit_quant"].unique()) >= 2:
            aux_lable_name += "-" + filter_df.iloc[index]["bit_quant"]
        model_name = filter_df.iloc[index]["name"] + aux_lable_name
        model_accuracy_info_dict[model_name] = {
            "model_tar": filter_df.iloc[index]["model_tar_id"],
            "opt_dump": filter_df.iloc[index]["opt_dump_id"],
            "opt_cfg": filter_df.iloc[index]["opt_cfg_id"],
            "opt_log": filter_df.iloc[index]["opt_log_id"],
            "opt_json": filter_df.iloc[index]["opt_json_id"],
            "opt_txt": filter_df.iloc[index]["opt_txt_id"],
        }

    if opt_json_show and model_accuracy_info_dict:
        with st.container(border=True):
            col_opt_json = st.columns(len(model_accuracy_info_dict))
            for index, json in enumerate(model_accuracy_info_dict.keys()):
                with col_opt_json[index]:
                    st.header(json, divider='rainbow')
                    opt_json_dict = get_compiled_json_file(client, ObjectId(model_accuracy_info_dict[json]["opt_json"]))
                    with st.expander("OPT JSON"):
                        st.json(opt_json_dict)
                    opt_df_dict = {}
                    opt_df_dict["name"] = []
                    opt_df_dict["layer_id"] = []
                    opt_df_dict["similarity"] = []
                    opt_df_dict["mse"] = []
                    opt_df_dict["ts_name"] = []
                    opt_df_dict["ts_min"] = []
                    opt_df_dict["ts_max"] = []
                    for op, val in opt_json_dict.items():
                        info = val["just_for_display"]["brief_info"].replace(" ", "").strip("}{")
                        pattern=r"[0-9a-zA-Z_]+=(?:[0-9a-zA-Z.]+,){1,}|[0-9a-zA-Z_]+=(?:\[[0-9.]+\],){1,}"
                        info_list = re.findall(pattern, info)
                        layer_id = int(info_list[0].split("=")[1].split(",")[0])
                        similarity = float(info_list[2].split("=")[1].split(",")[0])
                        if len(info_list) ==4:
                            mse = float(info_list[3].split("=")[1].split(",")[0].strip("]["))
                        else:
                            mse = 0.0
                        opt_df_dict["name"].append(op)
                        opt_df_dict["layer_id"].append(layer_id)
                        opt_df_dict["similarity"].append(similarity)
                        opt_df_dict["mse"].append(mse)
                        ## quant info update
                        quant_info_dict = val["just_for_display"]["quantization_info"]
                        tensor_name, quant_info = quant_info_dict.split(":", 1)
                        tensor_name = str(tensor_name.strip("\'}{"))
                        quant_info = quant_info.replace(" ", "").strip("}{")
                        quant_list = quant_info.split(",")
                        val_min = float(quant_list[6].split(":")[1].replace(" ", "").strip("\']["))
                        val_max = float(quant_list[7].split(":")[1].replace(" ", "").strip("\']["))

                        opt_df_dict["ts_name"].append(tensor_name)
                        opt_df_dict["ts_min"].append(val_min)
                        opt_df_dict["ts_max"].append(val_max)

                    st.dataframe(opt_df_dict, use_container_width=True)

                    chart_data = pd.DataFrame(opt_df_dict)

                    fig_bar = px.bar(chart_data, x="layer_id", y="similarity",  title="Layers Similarity")
                    st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")

                    fig_mse_bar = px.bar(chart_data, x="layer_id", y="mse",  title="Layers MSE")
                    st.plotly_chart(fig_mse_bar, use_container_width=True, theme="streamlit")

                    val_range_fig_bar = px.bar(chart_data, x="layer_id", y=["ts_min", "ts_max"], title="Tensor Range")
                    st.plotly_chart(val_range_fig_bar, use_container_width=True, theme="streamlit")


    if opt_log_show and model_accuracy_info_dict:
        with st.container(border=True):
            col_opt_log = st.columns(len(model_accuracy_info_dict))
            for index, log in enumerate(model_accuracy_info_dict.keys()):
                with col_opt_log[index]:
                    st.header(log, divider='rainbow')
                    log_file = get_compiled_models_files(client,
                            ObjectId(model_accuracy_info_dict[log]["opt_log"]))
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        opt_log_path = os.path.join(tmpdirname, "opt.log")
                        with open(opt_log_path, 'wb') as f:
                            f.write(log_file.read())
                        with open(opt_log_path, 'r') as f:
                            st.text(f.read())


    if opt_cfg_show and model_accuracy_info_dict:
        with st.container(border=True):
            col_opt_cfg = st.columns(len(model_accuracy_info_dict))
            for index, cfg in enumerate(model_accuracy_info_dict.keys()):
                with col_opt_cfg[index]:
                    st.header(cfg, divider='rainbow')
                    cfg_file = get_compiled_models_files(client,
                            ObjectId(model_accuracy_info_dict[cfg]["opt_cfg"]))
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        opt_cfg_path = os.path.join(tmpdirname, "opt.cfg")
                        with open(opt_cfg_path, 'wb') as f:
                            f.write(cfg_file.read())
                        with open(opt_cfg_path, 'r') as f:
                            st.text(f.read())

    if opt_txt_show and model_accuracy_info_dict:
        with st.container(border=True):
            col_opt_txt = st.columns(len(model_accuracy_info_dict))
            for index, txt in enumerate(model_accuracy_info_dict.keys()):
                with col_opt_txt[index]:
                    st.header(txt, divider='rainbow')
                    txt_file = get_compiled_models_files(client,
                            ObjectId(model_accuracy_info_dict[txt]["opt_txt"]))
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        opt_txt_path = os.path.join(tmpdirname, "opt.txt")
                        with open(opt_txt_path, 'wb') as f:
                            f.write(txt_file.read())
                        with open(opt_txt_path, 'r') as f:
                            st.text(f.read())



#with st.sidebar:
#    selected_op_node = st.selectbox(
#        'op',
#        opt_df_dict["name"],
#        placeholder="Select OP...")
#st.bar_chart(chart_data, x="name", y="similarity")
#st.line_chart(chart_data, x="name", y="similarity")

#fig_box = go.Figure()
#data_list = []
#label_list = []

#float_tensor_dict = {
#    "output":[],
#}

#quant_tensor_dict = {
#    "output":[],
#}

#conv_weight = []

#tar = tarfile.open(opt_dump_tar_file, mode="r:gz")
#for t in tar.getmembers():
#    if selected_op_node in t.name:
#        print(t.name)
#        array_file = BytesIO()
#        array_file.write(tar.extractfile(t).read())
#        array_file.seek(0)
#        a = np.load(array_file)
#        print(a.shape)
#        data_list.append(a.flatten())
#        label_list.append(t.name)
#        if "float32" in t.name:
#            if "o0" in t.name:
#                float_tensor_dict["output"] = a.flatten()
#            elif "Convolution" in t.name and "convolutionweights" in t.name:
#                conv_weight = a
        #    if "i0" in t.name:
        #        float_tensor_dict["input"] = a.flatten()
        #    elif "_w_" in t.name:
        #        float_tensor_dict["weight"] = a.flatten()
#        elif "quant" in t.name:
#            if "o0" in t.name:
#                quant_tensor_dict["output"] = a.flatten()
        #    if "i0" in t.name:
        #        quant_tensor_dict["input"] = a.flatten()
        #    elif "_w_" in t.name:
        #        quant_tensor_dict["weight"] = a.flatten()
#if "convolution" in selected_op_node:
    #fig = plt.figure()
#    fig, ax = plt.subplots()
#    c = conv_weight.shape[0]
#    wc = conv_weight.reshape(c, -1)
    #for index in range(0,c):
#    ax.boxplot(wc, vert=True)
#    st.pyplot(fig)
#    fig, ax = plt.subplots()
#    ax.hist(conv_weight.flatten(), bins=255)
#    st.pyplot(fig)
#else:
#    for k,v in float_tensor_dict.items():
        #fig, ax = plt.subplots()
#        fig = plt.figure()
#        ax1 = fig.add_subplot(121)
#        ax1.hist(v, bins=255)
#        ax2 = fig.add_subplot(122)
#        ax2.boxplot(v, vert=False)
        #ax3 = fig.add_subplot(222)
        #ax3.hist(quant_tensor_dict[k], bins=255)
        #ax4 = fig.add_subplot(224)
        #ax4.boxplot(quant_tensor_dict[k], vert=False)
#        st.pyplot(fig)

#st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")
#fig_dist = ff.create_distplot(data_list, label_list, bin_size=.1)
#st.plotly_chart(fig_dist, use_container_width=True, theme="streamlit")
        #with tar.extractfile(t) as fileobj:
        #    d = np.load(fileobj)
        #    print(d)
        #    fo = open(fileobj.name, 'rb')
        #file_contents = tar.extractfile(t).read()
        #array = np.loadtxt(io.StringIO(file_contents))
        #print(array.shape)
        #print(type(data_npy))
        #print(dir(data_npy))
        #data_bytes = data_npy.read()
        #data = np.frombuffer(data_bytes, dtype=np.float32)
        #print(data.shape)
