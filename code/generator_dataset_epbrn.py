import pandas as pd
import numpy as np
import random
from numpy.random import choice
import math
import sys


def process_record(rc_input, df, all_error_types, all_error_weights, all_fields):
    no_error = np.random.poisson(1, 1)
    #print("no_error:", no_error)
    errortypes = choice(all_error_types, no_error, p=all_error_weights)
    rc = rc_input.copy()
    for errortype in errortypes:
        if errortype == 'abr':
            if len(rc["surname"])>0:
                #rc["surname"] = rc["surname"][0]
                rc["surname"] = rc["surname"][0]
        if errortype == 'jwd1':
            rc["surname"] = rc["surname"]+'-' +rc["given_name"]
            rc["given_name"] = ''
        if errortype == 'jwd2':
            rc["given_name"] = rc["surname"]+'-' +rc["given_name"]
            rc["surname"] = ''
        if errortype == 'jwb1':
            rc["surname"] = rc["surname"]+' ' +rc["given_name"]
            rc["given_name"] = ''
        if errortype == 'jwb2':
            rc["given_name"] = rc["surname"]+' ' +rc["given_name"]
            rc["surname"] = ''
        if errortype == 'drf':    
            selected_field = random.choice(all_fields)
            rc[selected_field] = ''
        if errortype == 'dlc1':
            if len(rc["surname"])>0:
                rc["surname"] = rc['surname'][0:-1]
        if errortype == 'dlc2':
            if len(rc["given_name"])>0:
                rc["given_name"] = rc['given_name'][0:-1]
        if errortype == 'swn':
            temp = rc['given_name']
            rc["given_name"] = rc['surname']
            rc["surname"] = temp
        if errortype == 'swd': 
            temp = rc['day']
            rc['day'] = rc['month']
            rc['month'] = temp
        if errortype == 'rsd':
            rc['day'] = '01'
            rc['month'] = '01'
        if errortype == 'chy': 
            if rc['year'] != 'NaT' and not pd.isna(rc['year']) and rc['year'] != '':
                margin = random.choice(range(-5,5))
                rc['year'] = str( int(rc['year']) + margin)
        if errortype == 'chz':
            if len(str(rc['postcode']))== 4:
                selected_digit = random.choice(range(4))
                code = list(str(rc['postcode']))
                code[selected_digit] = str( random.choice(range(9)))
                rc['postcode'] = int(''.join(code))
        if errortype == 'mar':
            rc["surname"] = df.iloc[random.choice(range(len(df)))]['surname']
        if errortype == 'twi':
            rc["given_name"] = df.iloc[random.choice(range(len(df)))]['given_name']
        if errortype == 'add':
            rc['address_1'] = df.iloc[random.choice(range(len(df)))]['address_1']
            rc['address_2'] = df.iloc[random.choice(range(len(df)))]['address_2']
            rc['street_number'] = random.choice(range(500))
    return rc


# Preprocess
def preprocess(inputfile):
    df = pd.read_csv(inputfile, parse_dates=["date_of_birth"])

    df["rec_id"] = range(len(df))
    df['rec_id'] = df['rec_id'].astype(str)
    df['day'] = df['date_of_birth'].dt.strftime('%d')
    df['month'] = df['date_of_birth'].dt.strftime('%m')
    df['year'] = df['date_of_birth'].dt.strftime('%Y')
    df['postcode'] =   df['postcode'].fillna('0000')
    df['postcode'] = df['postcode'].astype(int)
    df['street_number'] =   df['street_number'].fillna('0')
    df['street_number'] = df['street_number'].astype(int)
    df = df.drop(["age", "phone_number", "soc_sec_id", "blocking_number", "date_of_birth"], axis=1)
    for col in ["surname", "given_name", "address_1", "address_2", "day", "month"]:
        df[col] = df[col].fillna('')
        df[col] = df[col].astype(str)
    df["match_id"] = range(len(df))
    all_fields = df.columns.values.tolist()
    all_fields.remove('rec_id')

    print("Original dataset length:",len(df))
    df.head()
    return df, all_fields

def generate_links(df, inputfile, count_shared):
    # Generate random indices for linkages
    leng = len(df)
    print("Process", inputfile, ", total records:", leng, "...")
    no_double_linked = int(leng*count_shared[0]/100)
    no_triple_linked = int(leng*count_shared[1]/100)
    no_quad_linked = int(leng*count_shared[2]/100)
    list_double_linked = random.sample(range(leng),k=no_double_linked)
    remain = [item for item in range(leng) if item not in list_double_linked]
    list_triple_linked = random.sample(remain,k=no_triple_linked)
    remain = [item for item in remain if item not in list_triple_linked]
    list_quad_linked = random.sample(remain,k=no_quad_linked)
    print("Double links:", no_double_linked,". Triple links:",no_triple_linked,". Quad links:",no_quad_linked)
    print("Total records after generated:", leng + no_double_linked + no_triple_linked*2 + no_quad_linked*3)
    print("Matched pairs:", no_double_linked + no_triple_linked*3 + no_quad_linked*6)
    return list_double_linked, list_triple_linked, list_quad_linked



def generate_files(PATH_FILES, type_set):

    if type_set == "train":
        INPUT_FILE_NAME = "ePBRN_F_original.csv"
        OUTPUT_FILE_NAME = "ePBRN_F_rep.csv"
    elif type_set == "test":
        
        INPUT_FILE_NAME = "ePBRN_D_original.csv"
        OUTPUT_FILE_NAME = "ePBRN_D_rep.csv"

    # Input the percentage of  [2, 3, 4] shared records in one linkage:
    count_shared = [1.68+21.0659, 1.9986 + 0.0471, 0.05]

    # Assigning the weights for each type of error:

    abr = 1 # abbreviation on surname: Michael -> M
    jwd1 = 1 # join with dash: John Peter -> John-Peter, join surname and given name into surname
    jwd2 = 1 # join with dash: John Peter -> John-Peter, join surname and given name into given name
    jwb1 = 1 # join with blank: 
    jwb2 = 1 # join with blank: 
    drf = 1 # drop all tokens in any field
    dlc1 = 1 # drop last character in surname: Peter -> Pete
    dlc2 = 1 # drop last character in given name
    swn = 1 # swap surname and given name: John Peter -> Peter John
    #swc1 = 1 # swap character in surname: Peter -> Petre
    #swc2 = 1 # swap character in given name: Peter -> Petre
    swd = 1 # swap day and month fields: 12/04 -> 04/12
    rsd = 1 # reset day and month: 12/04/1991 -> 01/01/1991
    chy = 1 # change year of birth by a margin of (+/-)5 
    #drz1 = 1 # drop leading zeros from day of birth: 02/04 -> 2/04
    #drz2 = 1 # drop leading zeros from month of birth: 02/04 -> 02/4
    chz = 1 # change any number of digit from zip code
    mar = 1 # change the whole token of surname: Mary Ward -> Mary Winston
    twi = 1 # duplicate all fields except given name: Micheal Williams -> Leo Williams
    add = 1 # change the whole 3 fields of address by randomly replacing each field by any other row

    all_error_types = ['abr','jwd1','jwd2','jwb1','jwb2' ,'drf','dlc1','dlc2','swn',
                    'rsd','chy','chz','mar','twi','add']
    all_error_weights = [abr, jwd1, jwd2, jwb1, jwb2, drf, dlc1, dlc2, swn, rsd, chy, chz, mar, twi, add]
    all_error_weights = all_error_weights/sum(np.asarray(all_error_weights))


    inputpath = PATH_FILES + INPUT_FILE_NAME
    outputpath = PATH_FILES + OUTPUT_FILE_NAME

    df, all_fields = preprocess(inputpath)
    list_double_linked, list_triple_linked, list_quad_linked = generate_links(df, inputpath, count_shared)
        
    # Main steps
    df1 = df
    j = 0
    tmp = []
    
    for list_linked in [list_double_linked, list_triple_linked, list_quad_linked]:
        j = j + 1
        for i in list_linked:
            for k in range(j):
                record_to_process = df.iloc[i]
                processed_rc = process_record(record_to_process, df, all_error_types, all_error_weights, all_fields)
                processed_rc["rec_id"] = processed_rc["rec_id"] + "-dup-" + str(k)
                df1.loc[len(df1.index)] = processed_rc
                    
                    
    # Save to disk    
    df1.to_csv(outputpath, index=False)
    print("Saved to", outputpath)

    return outputpath


if __name__ == "__main__":
    
    PATH_FILES =  "../data/ePBRN/"
    generate_files(PATH_FILES, "train")
