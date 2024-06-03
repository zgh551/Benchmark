import os
import json
import pymongo
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
from gridfs import GridFS
from typing import Any, Optional
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


class MongoDB:
    def __init__(self):
        self._client = None

    @st.cache_resource(hash_funcs={"__main__.MongoDB": lambda x: hash(x._client)})
    def init_connection(self):
        self._client = pymongo.MongoClient(**st.secrets["mongo"])

    @st.cache_data(ttl=300)
    def get_models_benchmark(self,
            mod_name: Optional[str]=None,
            device: Optional[str]=None,
            sdk: Optional[str]=None,
            chip: Optional[str]=None,
            date: Optional[str]=None):
        find_dict = {}
        db = self._client.sdnn_benchmark
        if mod_name:
            find_dict["name"] = mod_name
            if device:
                find_dict["device"] = device
            if sdk:
                find_dict["sdk"] = sdk
            if chip:
                find_dict["chip"] = chip
            if date:
                find_dict["date"] = date
            items = db.performance.find(find_dict)
            items = list(items)  # make hashable for st.cache_data
            return items
        else:
            items = db.performance.find(find_dict)
            items = list(items)  # make hashable for st.cache_data
            return items

    @st.cache_data(ttl=300)
    def get_next_items(
            name: Optional[str]=None,
            device: Optional[str]=None,
            sdk: Optional[str]=None,
            chip: Optional[str]=None,
            date: Optional[str]=None):
        items_list = []
        if name:
            items = get_models(name, device, sdk, chip, date)
            for item in items:
                if name and device and sdk and chip and date:
                    continue
                elif name and device and sdk and chip:
                    if not item["date"] in items_list:
                        items_list.append(item["date"])
                elif name and device and sdk:
                    if not item["chip"] in items_list:
                        items_list.append(item["chip"])
                elif name and device:
                    if not item["sdk"] in items_list:
                        items_list.append(item["sdk"])
                else:
                    if not item["device"] in items_list:
                        items_list.append(item["device"])
        else:
            items = get_models()
            for item in items:
                if not item["name"] in items_list:
                    items_list.append(item["name"])
        return items_list


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo"])

# Pull data from the collection.
# Uses st.cache_data to only rerun when the query changes or after 5 min.
@st.cache_data(ttl=300)
def get_models_performance(
        _client,
        mod_name: Optional[str]=None,
        device: Optional[str]=None,
        sdk: Optional[list]=None,
        chip: Optional[list]=None,
        acc_level: Optional[list]=None,
        date: Optional[list]=None):
    find_dict = {}
    db = _client.sdnn_benchmark
    if mod_name:
        find_dict["name"] = mod_name
        if device:
            find_dict["device"] = device
        if sdk:
            find_dict["sdk"] = {"$in": sdk}
        if chip:
            find_dict["chip"] = {"$in": chip}
        if acc_level:
            find_dict["acc_level"] = {"$in": acc_level}
        if date:
            find_dict["date"] = {"$in": date}
        items = db.performance.find(find_dict)
        items = list(items)  # make hashable for st.cache_data
        return items
    else:
        items = db.performance.find(find_dict)
        items = list(items)  # make hashable for st.cache_data
        return items

#@st.cache_data(ttl=300)
def get_models_performance_files(_client, file_id):
    db = _client.sdnn_benchmark
    performance_gridfs = GridFS(db, 'performance')
    return performance_gridfs.get(file_id)


####### Accuracy Model #######
@st.cache_data(ttl=300)
def get_compiled_models(_client):
    db = _client.sdnn_model_zoo
    return list(db.compiled_models.find({"device": "aipu"}))

def get_compiled_models_files(_client, file_id):
    db = _client.sdnn_model_zoo
    compiled_gridfs = GridFS(db, 'compiled_models')
    return compiled_gridfs.get(file_id)

def get_compiled_json_file(client, file_id):
    json_file = get_compiled_models_files(client, file_id)
    return  json.load(json_file)
###############################################
@st.cache_data(ttl=60)
def get_next_items(
        _client,
        name: Optional[str]=None,
        device: Optional[str]=None,
        sdk: Optional[list]=None,
        chip: Optional[list]=None,
        acc_level: Optional[list]=None,
        date: Optional[list]=None):
    items_list = []
    if name:
        items = get_models(_client, name, device, sdk, chip, acc_level, date)
        for item in items:
            if name and device and sdk and chip and acc_level and date:
                continue
            elif name and device and sdk and chip and acc_level:
                if not item["date"] in items_list:
                    items_list.append(item["date"])
            elif name and device and sdk and chip:
                if not item["acc_level"] in items_list:
                    items_list.append(item["acc_level"])
            elif name and device and sdk:
                if not item["chip"] in items_list:
                    items_list.append(item["chip"])
            elif name and device:
                if not item["sdk"] in items_list:
                    items_list.append(item["sdk"])
            else:
                if not item["device"] in items_list:
                    items_list.append(item["device"])
    else:
        items = get_models(_client)
        for item in items:
            if not item["name"] in items_list:
                items_list.append(item["name"])
    return items_list

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    st.header("Filter", divider='rainbow')
    #modify = st.checkbox("Add filters")
    modify = st.toggle('Add filters')

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def get_shape(data: list):
    shape_str = ""
    for d in data:
        shape_str += "(%s)%s " % (d['dtype'], str(d['shape']))
    return shape_str

def get_mean(data: list):
    np_data = np.array(data)
    if len(np_data) == 0:
        return 0
    sorted_data = np.sort(np_data)
    median_index = np.median(sorted_data)
    if len(sorted_data) % 2 == 1:
        return sorted_data[int(median_index)]
    else:
        return (sorted_data[int(median_index)] + sorted_data[int(median_index) -1 ]) * 0.5

def get_npz_file(client, file_id):
    npz_file = get_models_performance_files(client, file_id)
    with tempfile.TemporaryDirectory() as tmpdirname:
        npz_file_path = os.path.join(tmpdirname, "file.npz")
        with open(npz_file_path, 'wb') as f:
            f.write(npz_file.read())
        return np.load(npz_file_path)

def get_json_file(client, file_id):
    deploy_json = get_models_performance_files(client, file_id)
    return  json.load(deploy_json)


