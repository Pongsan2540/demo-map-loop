import configparser
import sys
import os

def networks_set():

    config_set_up = configparser.ConfigParser()
    config_set_up.read('./config_tempesta/config_tempesta.conf')
    url_main = config_set_up['parameters-set-up'].get('url', None)

    if 'url' in config_set_up['parameters-set-up'] and url_main is not None and url_main != "" and str(url_main) != "False" :

        ip_address = url_main.split('//')[-1]

        change_config_main = configparser.ConfigParser()
        change_config_main.read('config_set_up_api_analyze.txt')

        if 'parameters-set-up' in change_config_main:
            change_config_main['parameters-set-up']['api_url'] = str(ip_address)

        if 'send-status' in change_config_main:
            change_config_main['send-status']['url_leader'] = str(ip_address)

        with open('config_set_up_api_analyze.txt', 'w') as configfile:
            change_config_main.write(configfile, space_around_delimiters=False)

        print("---------- Network type : Host ---------------")
    else :

        change_config_main = configparser.ConfigParser()
        change_config_main.read('config_set_up_api_analyze.txt')

        if 'parameters-set-up' in change_config_main:
            change_config_main['parameters-set-up']['api_url'] = 'tp-analyze'

        if 'send-status' in change_config_main:
            change_config_main['send-status']['url_leader'] = 'tp-status'

        with open('config_set_up_api_analyze.txt', 'w') as configfile:
            change_config_main.write(configfile, space_around_delimiters=False)

        print("---------- Network type : Docker ---------------")
