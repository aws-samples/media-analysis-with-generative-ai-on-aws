import os
import json
import math
from fractions import Fraction
from shutil import rmtree
import boto3

def save_to_file(output_file, data):
    with open(output_file, 'w', encoding='utf-8') as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, ensure_ascii=False)
    return output_file

def save_json_to_file(name, json_data):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)
    return json_data

def to_hhmmssms(milliseconds):
    hh = math.floor(milliseconds / 3600000)
    mm = math.floor((milliseconds % 3600000) / 60000)
    ss = math.floor((milliseconds % 60000) / 1000)
    ms = math.ceil(milliseconds % 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def to_fraction(s):
    if isinstance(s, str):
        return Fraction(s.replace(':', '/')) 
    return Fraction(s)

def smpte_to_milliseconds(smpte):
    return smpte['StartTimecode']['Seconds'] * 1000 + smpte['StartTimecode']['Milliseconds']

def seconds_to_milliseconds(seconds):
    return seconds * 1000
    
def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    

def rmdir(directory):
    if os.path.exists(directory):
        rmtree(directory)

def upload_object(bucket, prefix, file):
        
        key = os.path.join(prefix, file)

        s3_client = boto3.client('s3')

        with open(file, "rb") as f:
            response = s3_client.put_object(
                Body=f,
                Bucket=bucket,
                Key=key,
            )
        return {"Bucket":bucket, "Key":key}
    

